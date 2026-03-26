# detection/views.py
import os
import pickle
import numpy as np
from django.shortcuts import render
from django.http import JsonResponse
from django.conf import settings
from django.core.files.storage import FileSystemStorage

# For text analysis
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# For image analysis (OCR)
try:
    import pytesseract
    PYTESSERACT_AVAILABLE = True
except ImportError:
    PYTESSERACT_AVAILABLE = False
    print("⚠️ pytesseract not available")

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("⚠️ PIL not available")

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("⚠️ cv2 not available")

# Load your trained model
MODEL_PATH = os.path.join(settings.BASE_DIR, 'models', '../models/cyberbullying_model.pkl')
VECTORIZER_PATH = os.path.join(settings.BASE_DIR, 'models', '../models/tfidf_vectorizer.pkl')

# If models are in detection app folder
# MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'cyberbullying_model.pkl')

print(f"Looking for model at: {MODEL_PATH}")
print(f"Model file exists: {os.path.exists(MODEL_PATH)}")
print(f"Vectorizer file exists: {os.path.exists(VECTORIZER_PATH)}")

# Load model once when server starts
try:
    if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
        model = pickle.load(open(MODEL_PATH, 'rb'))
        vectorizer = pickle.load(open(VECTORIZER_PATH, 'rb'))
        print("✅ Model and vectorizer loaded successfully!")
    else:
        print("❌ Model files not found at specified paths")
        model = None
        vectorizer = None
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None
    vectorizer = None


def home(request):
    """Homepage with options"""
    return render(request, 'detection/home.html')


def analyze_text(request):
    """Analyze single text comment"""
    result = None
    error_msg = None
    
    if request.method == 'POST':
        comment = request.POST.get('comment', '').strip()
        
        if not comment:
            error_msg = "Please enter a comment"
        elif model is None or vectorizer is None:
            error_msg = "Model not loaded. Check server logs."
        else:
            try:
                processed = comment.lower().strip()
                comment_tfidf = vectorizer.transform([processed])
                prediction = model.predict(comment_tfidf)[0]
                probabilities = model.predict_proba(comment_tfidf)[0]
                
                bullying_prob = probabilities[1] * 100
                safe_prob = probabilities[0] * 100
                
                result = {
                    'comment': comment,
                    'prediction': 'BULLYING' if prediction == 1 else 'SAFE',
                    'is_bullying': bool(prediction == 1),
                    'bullying_probability': round(bullying_prob, 2),
                    'safe_probability': round(safe_prob, 2),
                    'confidence': round(max(bullying_prob, safe_prob), 2),
                    'risk_level': 'HIGH' if bullying_prob > 80 else 'MODERATE' if bullying_prob > 50 else 'LOW'
                }
                
            except Exception as e:
                error_msg = f"Prediction error: {str(e)}"
                import traceback
                traceback.print_exc()
    
    context = {'result': result}
    if error_msg:
        context['error'] = error_msg
    
    return render(request, 'detection/text_analysis.html', context)


# ------------------------------
# OCR FUNCTION (exactly like notebook)
# ------------------------------
def extract_text_from_image(image_path, debug_info):
    try:
        image = Image.open(image_path)

        img_array = np.array(image)

        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array

        denoised = cv2.fastNlMeansDenoising(gray)
        _, thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        text = pytesseract.image_to_string(thresh)

        debug_info.append(f"✅ OCR chars: {len(text)}")

        return text.strip()

    except Exception as e:
        debug_info.append(f"❌ OCR error: {e}")
        return None
# ------------------------------
# TEXT ANALYSIS (exactly like notebook)
# ------------------------------
def analyze_extracted_text(text, debug_info):
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    debug_info.append(f"📊 Lines found: {len(lines)}")

    bullying_count = 0
    total_score = 0
    flagged = []
    valid_lines = 0

    for i, line in enumerate(lines):
        if len(line) < 3:
            continue
        valid_lines += 1
        line_tfidf = vectorizer.transform([line.lower()])
        pred = model.predict(line_tfidf)[0]
        prob = model.predict_proba(line_tfidf)[0][1] * 100
        total_score += prob
        if pred == 1:
            bullying_count += 1
            flagged.append({
                'line_num': i+1,
                'text': line[:100],
                'probability': round(prob, 2)
            })

    if valid_lines == 0:
        return {
            'total_messages': 0,
            'bullying_count': 0,
            'bullying_ratio': 0,
            'avg_score': 0,
            'status': 'NO TEXT FOUND',
            'risk_level': 'UNKNOWN',
            'flagged_messages': []
        }

    ratio = (bullying_count / valid_lines) * 100
    avg_score = total_score / valid_lines

    if ratio > 30 or avg_score > 50:
        status = '🚨 CYBERBULLYING DETECTED'
        risk = 'HIGH'
    elif ratio > 10 or avg_score > 30:
        status = '⚠️ SUSPICIOUS'
        risk = 'MODERATE'
    else:
        status = '✅ SAFE'
        risk = 'LOW'

    return {
        'total_messages': valid_lines,
        'bullying_count': bullying_count,
        'bullying_ratio': ratio,
        'avg_score': avg_score,
        'status': status,
        'risk_level': risk,
        'flagged_messages': sorted(flagged, key=lambda x: x['probability'], reverse=True)[:5]
    }


# ------------------------------
# MAIN VIEW (uses notebook logic)
# ------------------------------
def analyze_image(request):
    result = None
    debug_info = []

    if request.method == 'POST' and request.FILES.get('image'):
        image = request.FILES['image']

        # Save uploaded image
        upload_dir = os.path.join(settings.BASE_DIR, 'static', 'uploads')
        os.makedirs(upload_dir, exist_ok=True)
        fs = FileSystemStorage(location=upload_dir)
        filename = fs.save(image.name, image)
        image_path = fs.path(filename)
        debug_info.append(f"✅ Saved: {filename}, Path: {image_path}")

        # OCR
        extracted_text = extract_text_from_image(image_path, debug_info)
        debug_info.append(f"📄 OCR Preview: {extracted_text[:200]}..." if extracted_text else "❌ No text extracted")

        if extracted_text and len(extracted_text.strip()) > 3:
            analysis = analyze_extracted_text(extracted_text, debug_info)
            result = {
                'image_url': f'/static/uploads/{filename}',
                'extracted_text': extracted_text[:500],
                'total_messages': analysis['total_messages'],
                'bullying_count': analysis['bullying_count'],
                'bullying_ratio': round(analysis['bullying_ratio'], 2),
                'avg_score': round(analysis['avg_score'], 2),
                'status': analysis['status'],
                'risk_level': analysis['risk_level'],
                'flagged_messages': analysis['flagged_messages'],
                'debug_info': debug_info
            }
        else:
            result = {
                'error': 'No text found in image. Try a clear screenshot with larger text.',
                'debug_info': debug_info
            }

    return render(request, 'detection/image_analysis.html', {'result': result})


# ------------------------------
# API ANALYSIS
# ------------------------------
def api_analyze(request):
    if request.method == 'POST':
        comment = request.POST.get('comment', '').strip()
        if comment and model and vectorizer:
            tfidf = vectorizer.transform([comment.lower()])
            pred = model.predict(tfidf)[0]
            prob = model.predict_proba(tfidf)[0][1] * 100
            return JsonResponse({
                'prediction': 'BULLYING' if pred==1 else 'SAFE',
                'bullying_probability': round(prob,2),
                'safe_probability': round(100-prob,2)
            })
    return JsonResponse({'error':'Invalid request'}, status=400)


# ------------------------------
# DEBUG STATUS
# ------------------------------
def debug_status(request):
    return JsonResponse({
        'pytesseract': True,
        'pil': True,
        'cv2': True,
        'model_loaded': model is not None,
        'model_path': str(MODEL_PATH),
        'model_exists': os.path.exists(MODEL_PATH),
        'base_dir': str(settings.BASE_DIR),
    })