import logging
import os
import sys
import tempfile

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

# Make EEG-FM-Bench importable
BENCH_PATH = 'D:/EEG-FM-Bench'
if BENCH_PATH not in sys.path:
    sys.path.insert(0, BENCH_PATH)

from backend.model_loader import load_eegpt, load_eegnet, load_neurogpt
from backend.preprocessor import PreprocessorPipeline
from backend.inference import InferenceEngine

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(name)s: %(message)s')
logger = logging.getLogger(__name__)

# ── paths ────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
FRONTEND_DIR = os.path.join(BASE_DIR, 'frontend')
EEGPT_PATH = os.path.join(MODELS_DIR, 'eegpt_unified_epoch_6.pt')
EEGNET_PATH = os.path.join(MODELS_DIR, 'eegnet_adhd_epoch_10.pt')
NEUROGPT_PATH = os.path.join(MODELS_DIR, 'neurogpt_adhd_epoch_7.pt')

# ── model loading ─────────────────────────────────────────────────────────────
logger.info("Loading models...")
eegpt_model = load_eegpt(EEGPT_PATH) if os.path.exists(EEGPT_PATH) else None
eegnet_model = load_eegnet(EEGNET_PATH) if os.path.exists(EEGNET_PATH) else None
neurogpt_model = load_neurogpt(NEUROGPT_PATH) if os.path.exists(NEUROGPT_PATH) else None


if eegpt_model is None:
    logger.warning("EEGPT model not loaded — running in MOCK mode for EEGPT")
if eegnet_model is None:
    logger.warning("EEGNet model not loaded — running in MOCK mode for EEGNet")
if neurogpt_model is None:
    logger.warning("NeuroGPT model not loaded — running in MOCK mode for NeuroGPT")

MOCK_MODE = eegpt_model is None and eegnet_model is None and neurogpt_model is None

preprocessor = PreprocessorPipeline()
engine = InferenceEngine()

# ── model metadata ────────────────────────────────────────────────────────────
MODEL_INFO = {
    'eegpt': {
        'name': 'EEG-GPT',
        'accuracy': 0.788,
        'auroc': 0.896,
        'balanced_acc': 0.808,
        'auc_pr': 0.866,
        'epochs': 11,
        'best_epoch': 6,
        'batch_size': 32,
        'description': 'Foundation model trained from scratch (no pretrained weights)',
    },
    'eegnet': {
        'name': 'EEGNet',
        'accuracy': 0.671,
        'auroc': 0.893,
        'balanced_acc': 0.705,
        'auc_pr': 0.811,
        'epochs': 15,
        'best_epoch': 10,
        'batch_size': 1024,
        'description': 'Lightweight CNN baseline for EEG classification',
    },
    'neurogpt': {
        'name': 'NeuroGPT',
        'accuracy': 0.945,
        'auroc': 0.995,
        'balanced_acc': 0.949,
        'auc_pr': 0.994,
        'epochs': 12,
        'best_epoch': 7,
        'batch_size': 32,
        'description': 'EEG Conformer + GPT foundation model (pretrained)',
    },
}

# Map model_type -> (loaded model, mock flag)
def _get_model(model_type: str):
    if model_type == 'eegpt':
        return eegpt_model
    elif model_type == 'eegnet':
        return eegnet_model
    elif model_type == 'neurogpt':
        return neurogpt_model
    return None


# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(title='EEG Analysis Dashboard API')

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_methods=['*'],
    allow_headers=['*'],
)

# Serve frontend at root
@app.get('/', include_in_schema=False)
async def serve_frontend():
    index_path = os.path.join(FRONTEND_DIR, 'index.html')
    if os.path.exists(index_path):
        return FileResponse(index_path, media_type='text/html')
    raise HTTPException(status_code=404, detail='Frontend not found')


# ── endpoints ─────────────────────────────────────────────────────────────────

@app.get('/health')
async def health():
    return {
        'status': 'ok',
        'models_loaded': {
            'eegpt': eegpt_model is not None,
            'eegnet': eegnet_model is not None,
            'neurogpt': neurogpt_model is not None,
        },
        'mock_mode': MOCK_MODE,
    }


@app.get('/model_info/{model_type}')
async def model_info(model_type: str):
    if model_type not in MODEL_INFO:
        raise HTTPException(status_code=404, detail=f'Unknown model: {model_type}')
    return MODEL_INFO[model_type]


@app.post('/predict')
async def predict(
    file: UploadFile = File(...),
    model_type: str = Form(...),
):
    """Single-model prediction. Kept for backward compatibility."""
    if model_type not in ('eegpt', 'eegnet', 'neurogpt'):
        raise HTTPException(status_code=400,
                            detail=f"model_type must be 'eegpt', 'eegnet', or 'neurogpt', got '{model_type}'")

    # Save upload to a temp file
    suffix = os.path.splitext(file.filename or '.csv')[1] or '.csv'
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        # Preprocess
        try:
            preproc_result = preprocessor.preprocess(tmp_path)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        except Exception as exc:
            logger.exception("Preprocessing failed")
            raise HTTPException(status_code=500, detail=f'Preprocessing error: {exc}')

        windows_tensor = preproc_result['windows_tensor']
        eeg_signal = preproc_result['eeg_signal']
        band_powers = preproc_result['band_powers']
        n_windows = preproc_result['n_windows']

        # Select model
        model = _get_model(model_type)
        mock_this = model is None

        # Run inference
        try:
            result = engine.predict(windows_tensor, model, model_type)
        except Exception as exc:
            logger.exception("Inference failed")
            raise HTTPException(status_code=500, detail=f'Inference error: {exc}')

        return {
            'prediction': result['prediction'],
            'confidence': result['confidence'],
            'reliability': result['reliability'],
            'window_predictions': result['window_predictions'],
            'eeg_signal': eeg_signal,
            'band_powers': band_powers,
            'model_used': model_type,
            'windows_analyzed': n_windows,
            'mock_mode': mock_this,
        }

    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


@app.post('/predict_all')
async def predict_all(file: UploadFile = File(...)):
    """
    Run all three models on the same preprocessed signal.
    Returns per-model results + ensemble (majority vote) decision.
    """
    # Save upload to a temp file
    suffix = os.path.splitext(file.filename or '.csv')[1] or '.csv'
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        # ── 1. Preprocess ONCE ────────────────────────────────────────────────
        try:
            preproc_result = preprocessor.preprocess(tmp_path)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        except Exception as exc:
            logger.exception("Preprocessing failed")
            raise HTTPException(status_code=500, detail=f'Preprocessing error: {exc}')

        windows_tensor = preproc_result['windows_tensor']
        eeg_signal = preproc_result['eeg_signal']
        band_powers = preproc_result['band_powers']
        n_windows = preproc_result['n_windows']

        # ── 2. Run inference on each model ────────────────────────────────────
        per_model = {}
        for model_type in ('neurogpt', 'eegpt', 'eegnet'):
            model = _get_model(model_type)
            mock_this = model is None
            try:
                result = engine.predict(windows_tensor, model, model_type)
                per_model[model_type] = {
                    'prediction': result['prediction'],
                    'confidence': result['confidence'],
                    'reliability': result['reliability'],
                    'window_predictions': result['window_predictions'],
                    'mock_mode': mock_this,
                    'model_info': MODEL_INFO.get(model_type, {}),
                }
            except Exception as exc:
                logger.exception(f"Inference failed for {model_type}")
                per_model[model_type] = {
                    'prediction': None,
                    'confidence': 0.0,
                    'reliability': 'Low',
                    'window_predictions': [],
                    'mock_mode': mock_this,
                    'error': str(exc),
                    'model_info': MODEL_INFO.get(model_type, {}),
                }

        # ── 3. Compute ensemble (majority vote across models) ─────────────────
        valid_preds = [m['prediction'] for m in per_model.values() if m['prediction']]
        n_models = len(valid_preds)

        if n_models == 0:
            ensemble_prediction = None
            ensemble_confidence = 0.0
            ensemble_reliability = 'Low'
            agreement_str = '0/0'
        else:
            adhd_votes = sum(1 for p in valid_preds if p == 'ADHD')
            control_votes = n_models - adhd_votes

            ensemble_prediction = 'ADHD' if adhd_votes >= control_votes else 'Control'

            # Average confidence ONLY among models that agree with the winner
            agreeing_confs = [
                m['confidence'] for m in per_model.values()
                if m['prediction'] == ensemble_prediction
            ]
            ensemble_confidence = sum(agreeing_confs) / len(agreeing_confs) if agreeing_confs else 0.0

            agreeing_count = max(adhd_votes, control_votes)
            agreement_str = f"{agreeing_count}/{n_models}"

            # Reliability: full agreement + high conf = High
            if agreeing_count == n_models and ensemble_confidence >= 80:
                ensemble_reliability = 'High'
            elif agreeing_count == n_models:
                ensemble_reliability = 'Medium'
            elif agreeing_count >= 2:
                ensemble_reliability = 'Medium'
            else:
                ensemble_reliability = 'Low'

        return {
            'models': per_model,
            'ensemble': {
                'prediction': ensemble_prediction,
                'confidence': round(ensemble_confidence, 1),
                'reliability': ensemble_reliability,
                'agreement': agreement_str,
                'n_models': n_models,
            },
            'eeg_signal': eeg_signal,
            'band_powers': band_powers,
            'windows_analyzed': n_windows,
        }

    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
