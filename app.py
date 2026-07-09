# ------------------------------------------------------------------------
# NAFNet Restoration Studio - Flask backend
# ------------------------------------------------------------------------
import os
import io
import uuid
import zipfile
import traceback
from datetime import datetime

import cv2
import numpy as np
from flask import Flask, request, jsonify, render_template, send_file, session

import inference

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, 'static', 'uploads')
RESULT_DIR = os.path.join(BASE_DIR, 'static', 'results')
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

ALLOWED_EXT = {'.png', '.jpg', '.jpeg', '.bmp', '.webp'}
MAX_SIDE = 1600  # safety cap so CPU inference stays reasonably fast

app = Flask(__name__)
app.secret_key = os.environ.get('NAFNET_SECRET_KEY', 'dev-secret-change-me')

# Simple in-memory history (per server process). Swap for a DB if you need
# it to persist across restarts / multiple workers.
HISTORY = []


def _allowed(filename):
    return os.path.splitext(filename.lower())[1] in ALLOWED_EXT


def _read_image(file_storage):
    data = np.frombuffer(file_storage.read(), np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError('Could not decode image (unsupported or corrupt file).')
    h, w = img.shape[:2]
    if max(h, w) > MAX_SIDE:
        scale = MAX_SIDE / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return img


def _save_png(img, prefix):
    name = f"{prefix}_{uuid.uuid4().hex[:10]}.png"
    path = os.path.join(RESULT_DIR, name)
    cv2.imwrite(path, img)
    return name


def _process(img, task, width, low_light_pre, upscale_post, scale):
    """Runs the requested pipeline and returns (result_bgr, meta)."""
    meta = {'steps': []}

    working = img
    if low_light_pre:
        working = inference.enhance_low_light(working)
        meta['steps'].append('low_light_enhance (classical CLAHE + gamma)')

    if task in ('denoise', 'deblur'):
        working, elapsed = inference.run_nafnet(working, task, width, device='cpu')
        meta['steps'].append(f'{task} (NAFNet width{width}, {elapsed:.2f}s)')
    elif task == 'lowlight_only':
        if not low_light_pre:
            working = inference.enhance_low_light(working)
            meta['steps'].append('low_light_enhance (classical CLAHE + gamma)')
    elif task == 'upscale':
        pass  # handled below via upscale_post, task alone does nothing extra
    else:
        raise ValueError(f'Unknown task: {task}')

    if upscale_post or task == 'upscale':
        working = inference.upscale_and_sharpen(working, scale=scale)
        meta['steps'].append(f'upscale x{scale} (classical Lanczos + unsharp mask)')

    return working, meta


@app.route('/')
def index():
    return render_template('index.html', weight_status=inference.weight_status())


@app.route('/api/weights')
def api_weights():
    return jsonify(inference.weight_status())


@app.route('/api/process', methods=['POST'])
def api_process():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    if not file.filename or not _allowed(file.filename):
        return jsonify({'error': 'Unsupported file type'}), 400

    task = request.form.get('task', 'denoise')
    width = int(request.form.get('width', 32))
    low_light_pre = request.form.get('low_light_pre') == 'true'
    upscale_post = request.form.get('upscale_post') == 'true'
    scale = int(request.form.get('scale', 2))

    try:
        img = _read_image(file)
        original_name = _save_png(img, 'input')

        result, meta = _process(img, task, width, low_light_pre, upscale_post, scale)
        result_name = _save_png(result, 'output')

        entry = {
            'id': uuid.uuid4().hex[:8],
            'task': task,
            'width': width,
            'input_url': f'/static/results/{original_name}',
            'output_url': f'/static/results/{result_name}',
            'steps': meta['steps'],
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        }
        HISTORY.insert(0, entry)
        HISTORY[:] = HISTORY[:50]  # cap history size

        return jsonify({'ok': True, 'result': entry})

    except FileNotFoundError as e:
        return jsonify({'error': str(e), 'missing_weight': True}), 409
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/process_batch', methods=['POST'])
def api_process_batch():
    files = request.files.getlist('images')
    if not files:
        return jsonify({'error': 'No images uploaded'}), 400

    task = request.form.get('task', 'denoise')
    width = int(request.form.get('width', 32))
    low_light_pre = request.form.get('low_light_pre') == 'true'
    upscale_post = request.form.get('upscale_post') == 'true'
    scale = int(request.form.get('scale', 2))

    results = []
    errors = []
    batch_id = uuid.uuid4().hex[:8]

    for f in files:
        if not f.filename or not _allowed(f.filename):
            errors.append({'file': f.filename, 'error': 'unsupported file type'})
            continue
        try:
            img = _read_image(f)
            result, meta = _process(img, task, width, low_light_pre, upscale_post, scale)
            result_name = _save_png(result, f'batch_{batch_id}')
            entry = {
                'id': uuid.uuid4().hex[:8],
                'file': f.filename,
                'output_url': f'/static/results/{result_name}',
                'output_file': result_name,
                'steps': meta['steps'],
            }
            results.append(entry)
            HISTORY.insert(0, {**entry, 'task': task, 'width': width,
                                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')})
        except Exception as e:
            errors.append({'file': f.filename, 'error': str(e)})

    HISTORY[:] = HISTORY[:50]
    return jsonify({'ok': True, 'batch_id': batch_id, 'results': results, 'errors': errors})


@app.route('/api/download_batch/<batch_id>')
def download_batch(batch_id):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as zf:
        for fname in os.listdir(RESULT_DIR):
            if fname.startswith(f'batch_{batch_id}'):
                zf.write(os.path.join(RESULT_DIR, fname), fname)
    buf.seek(0)
    return send_file(buf, mimetype='application/zip', as_attachment=True,
                      download_name=f'nafnet_batch_{batch_id}.zip')


@app.route('/api/history')
def api_history():
    return jsonify(HISTORY[:50])


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
