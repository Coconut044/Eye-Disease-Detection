import streamlit as st
import tensorflow as tf
import cv2
import numpy as np
from PIL import Image
import io
from datetime import datetime
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.units import inch
import base64
import uuid

# Initialize eye detection cascade
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Set page configuration and mobile-responsive CSS
st.markdown("""
    <style>
        /* Mobile-first approach */
        .main {
            padding: 0 !important;
            max-width: 100% !important;
        }
        
        /* Header styling */
        .header-container {
            background: linear-gradient(90deg, #2563eb, #1d4ed8);
            padding: 1.5rem 1rem;
            text-align: center;
            color: white;
            margin-bottom: 1rem;
            border-radius: 0;
        }
        
        .header-title {
            font-size: 1.5rem;
            font-weight: bold;
            margin-bottom: 0.5rem;
            color: white;
            line-height: 1.2;
        }
        
        .header-subtitle {
            font-size: 1rem;
            color: #bfdbfe;
            margin-bottom: 0;
        }
        
        /* Tab styling */
        .stTabs {
            background-color: white;
            padding: 1rem;
            border-radius: 15px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            margin: 1rem auto;
            max-width: 100%;
        }
        
        .stTabs [data-baseweb="tab-list"] {
            gap: 10px;
            justify-content: center;
            background-color: #f8f9fa;
            padding: 0.5rem;
            border-radius: 10px;
            margin-bottom: 1rem;
            flex-wrap: wrap;
        }
        
        .stTabs [data-baseweb="tab"] {
            height: auto;
            width: calc(50% - 10px);
            min-width: 140px;
            background-color: white;
            border-radius: 10px;
            padding: 0.5rem;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            gap: 0.5rem;
            cursor: pointer;
            transition: all 0.3s ease;
            border: 2px solid transparent;
            font-size: 1rem !important;
        }
        
        .stTabs [data-baseweb="tab"]:hover {
            transform: translateY(-3px);
            box-shadow: 0 4px 12px rgba(37,99,235,0.2);
            border-color: #2563eb;
        }
        
        .stTabs [aria-selected="true"] {
            background-color: #EBF5FF !important;
            border-color: #2563eb !important;
            color: #2563eb !important;
            font-weight: bold;
        }
        
        /* Metrics container */
        .metrics-container {
            display: grid;
            grid-template-columns: 1fr;
            gap: 1rem;
            margin: 1rem 0;
            padding: 0 1rem;
        }
        
        .metric-card {
            background-color: #EBF5FF;
            padding: 1rem;
            border-radius: 12px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            text-align: center;
            transition: all 0.3s ease;
            border: 2px solid transparent;
        }
        
        .metric-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 8px 15px rgba(37,99,235,0.2);
            border-color: #2563eb;
            background-color: #F0F7FF;
        }
        
        .metric-title {
            font-size: 1rem;
            color: #4b5563;
            margin-bottom: 0.8rem;
            font-weight: 600;
        }
        
        .metric-value {
            font-size: 1.5rem;
            font-weight: bold;
            color: #2563eb;
        }
        
        /* Camera container */
        .camera-container {
            width: 100%;
            height: auto;
            max-width: 400px;
            margin: 0 auto;
        }
        
        .stCamera > div {
            width: 100% !important;
            height: auto !important;
        }
        
        .stCamera video {
            width: 100% !important;
            height: auto !important;
            object-fit: cover;
        }
        
        /* Notification styling */
        .notification {
            position: fixed;
            top: auto;
            bottom: 20px;
            right: 10px;
            left: 10px;
            padding: 0.8rem;
            border-radius: 8px;
            font-size: 0.9rem;
            z-index: 1000;
            animation: slideIn 0.3s ease-out;
            margin-bottom: 10px;
        }
        
        @keyframes slideIn {
            from {
                transform: translateY(100%);
                opacity: 0;
            }
            to {
                transform: translateY(0);
                opacity: 1;
            }
        }
        
        /* Button styling */
        .stButton > button {
            background-color: #2563eb;
            color: white;
            border-radius: 8px;
            padding: 0.6rem 1rem;
            font-size: 1rem;
            font-weight: 500;
            border: none;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            transition: all 0.3s ease;
            width: 100%;
        }
        
        .stButton > button:hover {
            background-color: #1d4ed8;
            transform: translateY(-2px);
            box-shadow: 0 6px 8px rgba(0,0,0,0.2);
        }
        
        /* Report container */
        .report-container {
            background-color: white;
            padding: 1rem;
            border-radius: 12px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            margin-top: 1rem;
        }
        
        /* Status indicators */
        .status-indicator {
            background-color: #f8f9fa;
            padding: 0.8rem;
            border-radius: 8px;
            text-align: center;
            margin-bottom: 0.8rem;
            font-size: 0.9rem;
        }
        
        .status-indicator.success {
            background-color: #d4edda;
            color: #155724;
        }
        
        .status-indicator.warning {
            background-color: #fff3cd;
            color: #856404;
        }
        
        .status-indicator.error {
            background-color: #f8d7da;
            color: #721c24;
        }
        
        /* File uploader */
        .stFileUploader {
            width: 100% !important;
            max-width: 100% !important;
        }
        
        /* Sidebar */
        @media (max-width: 768px) {
            .css-1d391kg {
                width: 100% !important;
            }
            
            .css-1v3fvcr {
                padding: 1rem;
            }
        }
        
        /* Responsive images */
        img {
            max-width: 100%;
            height: auto;
        }
        
        /* Responsive tables */
        table {
            width: 100%;
            overflow-x: auto;
            display: block;
        }
        
        /* Media Queries for larger screens */
        @media (min-width: 768px) {
            .header-title {
                font-size: 2.8rem;
            }
            
            .header-subtitle {
                font-size: 1.4rem;
            }
            
            .metrics-container {
                grid-template-columns: repeat(3, 1fr);
            }
            
            .stTabs [data-baseweb="tab"] {
                width: 300px;
            }
            
            .notification {
                width: auto;
                right: 20px;
                left: auto;
                bottom: 40px;
            }
            
            .stButton > button {
                width: auto;
            }
            
            .report-container {
                padding: 2rem;
            }
        }
    </style>
""", unsafe_allow_html=True)

def load_model():
    """Load the trained keras model"""
    try:
        model = tf.keras.models.load_model(r"best_eye_disease_model.keras")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def detect_eyes(image):
    """Detect eyes in the image and return detection status"""
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    eyes = eye_cascade.detectMultiScale(gray, 1.3, 5)
    
    return {'eyes_detected': len(eyes) > 0}

def preprocess_image(image):
    """Preprocess the image for model prediction"""
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    
    image = cv2.resize(image, (224, 224))
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    return image

def analyze_image_quality(image):
    """Analyze image quality and return notifications"""
    notifications = []
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    brightness = np.mean(gray)
    if brightness < 100:
        notifications.append({"message": "⚠️ Not enough light - Please increase lighting", "type": "warning"})
    elif brightness > 200:
        notifications.append({"message": "⚠️ Too much light - Please reduce lighting", "type": "warning"})
    
    contrast = np.std(gray)
    if contrast < 30:
        notifications.append({"message": "⚠️ Low contrast - Please adjust lighting for better clarity", "type": "warning"})
    
    return notifications

def display_notifications(notifications):
    """Display real-time notifications"""
    for i, notification in enumerate(notifications):
        st.markdown(f"""
            <div class="notification" style="background-color: {'#fff3cd' if notification['type'] == 'warning' else '#EBF5FF'};">
                {notification['message']}
            </div>
        """, unsafe_allow_html=True)

def get_prediction(model, image):
    """Get prediction from model"""
    try:
        prediction_proba = model.predict(image)[0][0]
        prediction = "Infected" if prediction_proba > 0.5 else "Healthy"
        confidence = prediction_proba if prediction == "Infected" else 1 - prediction_proba
        return prediction, confidence
    except Exception as e:
        st.error(f"Error making prediction: {e}")
        return None, None

def create_download_link(pdf_bytes, filename):
    """Create a download link for the PDF file"""
    b64 = base64.b64encode(pdf_bytes).decode()
    return f'<a href="data:application/pdf;base64,{b64}" download="{filename}">Download PDF Report</a>'

def generate_pdf_report(prediction, confidence, details, report_id, timestamp):
    """Generate a PDF report with the analysis results"""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=72)
    
    story = []
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        spaceAfter=30
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=14,
        spaceAfter=12,
        textColor=colors.HexColor('#2563eb')
    )
    
    story.append(Paragraph("Medical Analysis Report", title_style))
    story.append(Spacer(1, 20))
    
    metadata = [
        ["Report ID:", report_id],
        ["Date & Time:", timestamp],
        ["Analysis Type:", "Ophthalmic Examination"],
        ["Primary Assessment:", prediction],
        ["Confidence Level:", f"{confidence:.2%}"],
        ["Risk Level:", details["risk_level"]],
        ["Urgency:", details["urgency"]]
    ]
    
    meta_table = Table(metadata, colWidths=[2*inch, 4*inch])
    meta_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
        ('TOPPADDING', (0, 0), (-1, -1), 12),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 20))
    
    story.append(Paragraph("Clinical Observations", heading_style))
    for symptom in details["symptoms"]:
        story.append(Paragraph(f"• {symptom}", styles["Normal"]))
    story.append(Spacer(1, 20))
    
    story.append(Paragraph("Recommendations", heading_style))
    for recommendation in details["recommendations"]:
        story.append(Paragraph(f"• {recommendation}", styles["Normal"]))
    story.append(Spacer(1, 20))
    
    story.append(Paragraph("Technical Information", heading_style))
    story.append(Paragraph("AI Model: Eye Disease Detection System v1.0", styles["Normal"]))
    story.append(Paragraph("Analysis Protocol: Deep Learning Image Analysis", styles["Normal"]))
    story.append(Paragraph(f"Image Quality: {'Acceptable for Analysis' if confidence > 0.7 else 'Further Imaging Recommended'}", styles["Normal"]))
    
    story.append(Spacer(1, 30))
    disclaimer_style = ParagraphStyle(
        'Disclaimer',
        parent=styles['Normal'],
        fontSize=8,
        textColor=colors.grey
    )
    story.append(Paragraph("DISCLAIMER: This AI-generated report is for informational purposes only and should not be considered as a substitute for professional medical advice. Please consult with a qualified healthcare provider for proper medical evaluation and treatment.", disclaimer_style))
    
    doc.build(story)
    pdf_bytes = buffer.getvalue()
    buffer.close()
    
    return pdf_bytes

def modify_generate_report(prediction, confidence, image):
    """Modified version of generate_report that includes PDF download"""
    report_id = f"RPT-{uuid.uuid4().hex[:8].upper()}"
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
    
    condition_details = {
        "Infected": {
            "symptoms": [
                "Redness and inflammation",
                "Discharge or tearing",
                "Blurred vision",
                "Light sensitivity",
                "Eye pain or discomfort"
            ],
            "recommendations": [
                "Immediate consultation with an ophthalmologist is strongly advised",
                "Avoid touching or rubbing the affected eye",
                "Use prescribed eye drops or medications as directed",
                "Maintain proper eye hygiene",
                "Follow up within 48-72 hours to monitor progression"
            ],
            "risk_level": "High",
            "urgency": "Urgent medical attention recommended"
        },
        "Healthy": {
            "symptoms": [
                "No visible abnormalities",
                "Clear cornea and conjunctiva",
                "Normal eye appearance",
                "No signs of infection or inflammation"
            ],
            "recommendations": [
                "Continue regular eye check-ups",
                "Maintain good eye hygiene",
                "Use proper eye protection when needed",
                "Follow a healthy diet rich in vitamins A and C",
                "Take regular breaks during screen time (20-20-20 rule)"
            ],
            "risk_level": "Low",
            "urgency": "Routine follow-up recommended"
        }
    }
    
    details = condition_details.get(prediction, condition_details["Healthy"])
    pdf_bytes = generate_pdf_report(prediction, confidence, details, report_id, timestamp)
    
    symptoms_html = "".join([f"<li>{symptom}</li>" for symptom in details["symptoms"]])
    recommendations_html = "".join([f"<li>{rec}</li>" for rec in details["recommendations"]])
    
    report_html = f"""
    <div class="report-container">
        <h2>Medical Analysis Report</h2>
        
        <p><strong>Time:</strong> {timestamp}</p>
        <p><strong>Report ID:</strong> {report_id}</p>
        <p><strong>Analysis Type:</strong> Ophthalmic Examination</p>
        
        <h3>Diagnostic Results</h3>
        <p><strong>Primary Assessment:</strong> {prediction}</p>
        <p><strong>Confidence Level:</strong> {confidence:.2%}</p>
        <p><strong>Risk Level:</strong> {details["risk_level"]}</p>
        <p><strong>Urgency:</strong> {details["urgency"]}</p>
        
        <h3>Clinical Observations</h3>
        <p><strong>Observed Symptoms/Characteristics:</strong></p>
        <ul>
            {symptoms_html}
        </ul>
        
        <h3>Recommendations</h3>
        <ul>
            {recommendations_html}
        </ul>
        
        <h3>Technical Information</h3>
        <p><strong>AI Model:</strong> Eye Disease Detection System v1.0</p>
        <p><strong>Analysis Protocol:</strong> Deep Learning Image Analysis</p>
        <p><strong>Image Quality:</strong> {'Acceptable for Analysis' if confidence > 0.7 else 'Further Imaging Recommended'}</p>
        
        <div style="margin-top: 20px; padding: 10px; background-color: #f8f9fa; border-radius: 5px;">
            {create_download_link(pdf_bytes, f'eye_analysis_report_{report_id}.pdf')}
        </div>
    </div>
    """
    
    st.markdown(report_html, unsafe_allow_html=True)
    return pdf_bytes

def display_metrics(prediction, confidence):
    """Display metrics in responsive cards"""
    st.markdown(f"""
        <div class="metrics-container">
            <div class="metric-card">
                <h3 class="metric-title">Diagnosis</h3>
                <p class="metric-value">{prediction}</p>
            </div>
            <div class="metric-card">
                <h3 class="metric-title">Confidence</h3>
                <p class="metric-value">{confidence:.2%}</p>
            </div>
            <div class="metric-card">
                <h3 class="metric-title">Analysis Status</h3>
                <p class="metric-value">Complete ✓</p>
            </div>
        </div>
    """, unsafe_allow_html=True)

class ReportState:
    """Class to maintain report state"""
    def __init__(self):
        self.pdf_bytes = None
        self.report_id = None
        self.timestamp = None

if 'report_state' not in st.session_state:
    st.session_state.report_state = ReportState()

def main():
    model = load_model()
    if model is None:
        st.error("System Error: Model initialization failed. Please contact technical support.")
        return
    
    st.markdown("""
        <div class="header-container">
            <h1 class="header-title">Conjunctivitis Eye Disease Detection System</h1>
            <p class="header-subtitle">Advanced AI-Powered Optical Health Analysis</p>
        </div>
    """, unsafe_allow_html=True)
    
    with st.sidebar:
        st.markdown("### System Information")
        st.info("AI Model: Eye Disease Detection v1.0")
        st.info("Last Updated: 2024")
        
        st.markdown("### Quick Guide")
        st.markdown("""
            1. Choose input method (Camera/Upload)
            2. Provide clear eye image
            3. Review AI analysis
            4. Download report if needed
        """)
        
        st.markdown("### Download Report")
        if st.session_state.report_state.pdf_bytes is not None:
            download_button = st.download_button(
                label="📥 Download PDF Report",
                data=st.session_state.report_state.pdf_bytes,
                file_name=f'eye_analysis_report_{st.session_state.report_state.report_id}.pdf',
                mime='application/pdf',
            )
            st.markdown(f"Report ID: {st.session_state.report_state.report_id}")
            st.markdown(f"Generated: {st.session_state.report_state.timestamp}")
        else:
            st.info("Complete the analysis to generate a downloadable report")
        
        st.markdown("### Support")
        st.markdown("📧 akshara.sharma2@s.amity.edu")
        st.markdown("☎️ +91 (971) 719-4402")
    
    st.markdown('<div style="max-width: 1200px; margin: 0 auto;">', unsafe_allow_html=True)
    tab1, tab2 = st.tabs([
        "📸 Live Capture Analysis",
        "📤 Image Upload Analysis"
    ])
    
    with tab1:
        st.markdown("<h3 style='text-align: center;'>Real-time Eye Analysis</h3>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center;'>Please ensure proper lighting and steady positioning.</p>", unsafe_allow_html=True)
        
        status_col1, status_col2 = st.columns(2)
        with status_col1:
            eye_status = st.empty()
        with status_col2:
            quality_status = st.empty()
        
        camera_image = st.camera_input("Capture Eye Image", key="camera")
        
        if camera_image is not None:
            process_image(camera_image, model, eye_status, quality_status)
    
    with tab2:
        st.markdown("<h3 style='text-align: center;'>Upload Analysis</h3>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center;'>Upload a clear, well-lit image of the eye for analysis.</p>", unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader("Select eye image for analysis", type=['jpg', 'jpeg', 'png'])
        
        if uploaded_file is not None:
            process_image(uploaded_file, model, st.empty(), st.empty())
    
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
        <div style="background-color: #f8d7da; padding: 1rem; border-radius: 8px; margin-top: 2rem;">
            <h4 style="color: #721c24;">⚠️ Medical Disclaimer</h4>
            <p style="color: #721c24;">
                This AI-powered system is designed to assist in the detection of eye diseases. Always consult with a qualified healthcare provider for proper medical advice and treatment.
            </p>
        </div>
    """, unsafe_allow_html=True)

def process_image(image_input, model, eye_status, quality_status):
    """Process the input image and generate analysis"""
    image = Image.open(image_input)
    image_array = np.array(image)
    
    detection_status = detect_eyes(image_array)
    
    eye_status.markdown(
        f"Eyes Detected: {'✅' if detection_status['eyes_detected'] else '❌'}"
    )
    
    if detection_status['eyes_detected']:
        notifications = analyze_image_quality(image_array)
        display_notifications(notifications)
        
        quality_status.markdown(
            "Image Quality: ✅" if not any(n['type'] == 'warning' for n in notifications)
            else "Image Quality: ⚠️"
        )
        
        if not any(n['type'] == 'warning' for n in notifications):
            processed_image = preprocess_image(image_array)
            prediction, confidence = get_prediction(model, processed_image)
            
            if prediction and confidence:
                display_metrics(prediction, confidence)
                pdf_data = modify_generate_report(prediction, confidence, image)
                
                st.session_state.report_state.pdf_bytes = pdf_data
                st.session_state.report_state.report_id = f"RPT-{uuid.uuid4().hex[:8].upper()}"
                st.session_state.report_state.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    else:
        st.warning("Please ensure both eyes are visible in the frame.")
        quality_status.markdown("Image Quality: ❌")
        st.session_state.report_state = ReportState()

if __name__ == "__main__":
    main()
