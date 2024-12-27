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

# Set page configuration

# Complete CSS styling
st.markdown("""
<style>
/* Enhanced Tab Styling with mobile responsiveness */
.stTabs {
    background-color: white;
    padding: 1rem;
    border-radius: 15px;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    margin: 1rem auto;
    max-width: 100%;
    width: 100%;
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
    min-height: 60px;
    width: calc(50% - 10px);
    max-width: 300px;
    background-color: white;
    border-radius: 10px;
    padding: 0.5rem;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 0.25rem;
    cursor: pointer;
    transition: all 0.3s ease;
    border: 2px solid transparent;
    font-size: 1rem !important;
}

@media (max-width: 768px) {
    .stTabs [data-baseweb="tab"] {
        width: 100%;
        max-width: none;
    }
}

/* Header styling with mobile responsiveness */
.header-container {
    background: linear-gradient(90deg, #2563eb, #1d4ed8);
    padding: 1.5rem 1rem;
    text-align: center;
    color: white;
    margin-bottom: 1rem;
    border-radius: 0;
}

.header-title {
    font-size: 2rem;
    font-weight: bold;
    margin-bottom: 0.5rem;
    color: white;
}

@media (max-width: 768px) {
    .header-title {
        font-size: 1.5rem;
    }
    .header-subtitle {
        font-size: 1rem;
    }
}

/* Enhanced KPI Cards styling with mobile responsiveness */
.metrics-container {
    display: flex;
    flex-wrap: wrap;
    gap: 1rem;
    margin: 1rem 0;
}

.metric-card {
    background-color: #EBF5FF;
    padding: 1rem;
    border-radius: 12px;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    text-align: center;
    flex: 1 1 calc(33.333% - 1rem);
    min-width: 200px;
    transition: all 0.3s ease;
}

@media (max-width: 768px) {
    .metric-card {
        flex: 1 1 100%;
    }
}

/* Camera container styling with mobile responsiveness */
.camera-container {
    position: relative;
    width: 100%;
    max-width: 400px;
    margin: 0 auto;
}

.stCamera > div {
    position: relative;
    width: 100% !important;
    height: auto !important;
    margin: 0 auto;
}

.stCamera video {
    width: 100% !important;
    height: auto !important;
    object-fit: cover;
}

/* Real-time notification styling with mobile responsiveness */
.notification {
    position: fixed;
    top: 100px;
    right: 10px;
    left: 10px;
    padding: 0.75rem;
    border-radius: 8px;
    background-color: #EBF5FF;
    color: #2563eb;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    z-index: 1000;
    animation: slideIn 0.3s ease-out;
    margin-bottom: 10px;
}

/* Report container styling with mobile responsiveness */
.report-container {
    background-color: white;
    padding: 1rem;
    border-radius: 12px;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    margin-top: 1rem;
    width: 100%;
}

/* Status indicators styling */
.status-indicator {
    background-color: #f8f9fa;
    padding: 0.75rem;
    border-radius: 8px;
    text-align: center;
    margin-bottom: 0.75rem;
    width: 100%;
}

/* Make images responsive */
img {
    max-width: 100%;
    height: auto;
}

/* Make file uploader responsive */
.stFileUploader {
    width: 100%;
}

/* Make buttons responsive */
.stButton > button {
    width: 100%;
    margin-bottom: 0.5rem;
}

/* Make text inputs responsive */
.stTextInput > div {
    width: 100%;
}

/* Responsive table styling */
table {
    width: 100%;
    overflow-x: auto;
    display: block;
}

/* Make columns stack on mobile */
@media (max-width: 768px) {
    .st-emotion-cache-1l269bu {
        flex-direction: column;
    }
    .st-emotion-cache-1l269bu > * {
        width: 100%;
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
    """
    Detect eyes in the image and return detection status
    """
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    
    # Convert to grayscale for detection
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Detect eyes
    eyes = eye_cascade.detectMultiScale(gray, 1.3, 5)
    
    return {'eyes_detected': len(eyes) > 0}
    
    return detection_status

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
    
    # Convert image to grayscale for analysis
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Check brightness
    brightness = np.mean(gray)
    if brightness < 100:
        notifications.append({"message": "⚠️ Not enough light - Please increase lighting", "type": "warning"})
    elif brightness > 200:
        notifications.append({"message": "⚠️ Too much light - Please reduce lighting", "type": "warning"})
    
    # Check contrast
    contrast = np.std(gray)
    if contrast < 30:
        notifications.append({"message": "⚠️ Low contrast - Please adjust lighting for better clarity", "type": "warning"})
    
    return notifications

def display_notifications(notifications):
    """Display real-time notifications"""
    for i, notification in enumerate(notifications):
        st.markdown(f"""
            <div class="notification" style="background-color: {'#fff3cd' if notification['type'] == 'warning' else '#EBF5FF'}; top: {400 + i*70}px;">
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
    
    # Create story list to hold content
    story = []
    styles = getSampleStyleSheet()
    
    # Custom styles
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
    
    # Add title
    story.append(Paragraph("Medical Analysis Report", title_style))
    story.append(Spacer(1, 20))
    
    # Add metadata
    metadata = [
        ["Report ID:", report_id],
        ["Date & Time:", timestamp],
        ["Analysis Type:", "Ophthalmic Examination"],
        ["Primary Assessment:", prediction],
        ["Confidence Level:", f"{confidence:.2%}"],
        ["Risk Level:", details["risk_level"]],
        ["Urgency:", details["urgency"]]
    ]
    
    # Create metadata table
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
    
    # Add Clinical Observations
    story.append(Paragraph("Clinical Observations", heading_style))
    for symptom in details["symptoms"]:
        story.append(Paragraph(f"• {symptom}", styles["Normal"]))
    story.append(Spacer(1, 20))
    
    # Add Recommendations
    story.append(Paragraph("Recommendations", heading_style))
    for recommendation in details["recommendations"]:
        story.append(Paragraph(f"• {recommendation}", styles["Normal"]))
    story.append(Spacer(1, 20))
    
    # Add Technical Information
    story.append(Paragraph("Technical Information", heading_style))
    story.append(Paragraph("AI Model: Eye Disease Detection System v1.0", styles["Normal"]))
    story.append(Paragraph("Analysis Protocol: Deep Learning Image Analysis", styles["Normal"]))
    story.append(Paragraph(f"Image Quality: {'Acceptable for Analysis' if confidence > 0.7 else 'Further Imaging Recommended'}", styles["Normal"]))
    
    # Add disclaimer
    story.append(Spacer(1, 30))
    disclaimer_style = ParagraphStyle(
        'Disclaimer',
        parent=styles['Normal'],
        fontSize=8,
        textColor=colors.grey
    )
    story.append(Paragraph("DISCLAIMER: This AI-generated report is for informational purposes only and should not be considered as a substitute for professional medical advice. Please consult with a qualified healthcare provider for proper medical evaluation and treatment.", disclaimer_style))
    
    # Build PDF
    doc.build(story)
    pdf_bytes = buffer.getvalue()
    buffer.close()
    
    return pdf_bytes

def modify_generate_report(prediction, confidence, image):
    """Modified version of generate_report that includes PDF download"""
    # Generate unique report ID and timestamp
    report_id = f"RPT-{uuid.uuid4().hex[:8].upper()}"
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
    
    # Define condition-specific details (same as before)
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
    
    # Generate PDF
    pdf_bytes = generate_pdf_report(prediction, confidence, details, report_id, timestamp)
    
    # Display the HTML report as before
    symptoms_html = "".join([f"<li>{symptom}</li>" for symptom in details["symptoms"]])
    recommendations_html = "".join([f"<li>{rec}</li>" for rec in details["recommendations"]])
    
    report_html = f"""
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
    """
    
    st.markdown(report_html, unsafe_allow_html=True)

def display_metrics(prediction, confidence):
    """Display metrics in horizontal cards"""
    st.markdown("""
        <div class="metrics-container">
            <div class="metric-card">
                <h3 class="metric-title">Diagnosis</h3>
                <p class="metric-value">{}</p>
            </div>
            <div class="metric-card">
                <h3 class="metric-title">Confidence</h3>
                <p class="metric-value">{:.2%}</p>
            </div>
            <div class="metric-card">
                <h3 class="metric-title">Analysis Status</h3>
                <p class="metric-value">Complete ✓</p>
            </div>
        </div>
    """.format(prediction, confidence), unsafe_allow_html=True)

class ReportState:
    """Class to maintain report state"""
    def __init__(self):
        self.pdf_bytes = None
        self.report_id = None
        self.timestamp = None

# Initialize report state in session state
if 'report_state' not in st.session_state:
    st.session_state.report_state = ReportState()

def main():
    # Load model
    model = load_model()
    if model is None:
        st.error("System Error: Model initialization failed. Please contact technical support.")
        return
    
    # Header Section
    st.markdown("""
        <div class="header-container">
            <h1 class="header-title">Conjunctivitis Eye Disease Detection System</h1>
            <p class="header-subtitle">Advanced AI-Powered Optical Health Analysis</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Sidebar configuration
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
        
        # Download section in sidebar
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
    
    # Enhanced tabs with centered content
    st.markdown('<div style="max-width: 1200px; margin: 0 auto;">', unsafe_allow_html=True)
    tab1, tab2 = st.tabs([
        "📸 Live Capture Analysis",
        "📤 Image Upload Analysis"
    ])
    
    with tab1:
        st.markdown("<h3 style='text-align: center;'>Real-time Eye Analysis</h3>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center;'>Please ensure proper lighting and steady positioning.</p>", unsafe_allow_html=True)
        
        # Simplified status indicator
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

    # Footer with disclaimer
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
    
    # Perform eye detection
    detection_status = detect_eyes(image_array)
    
    # Update status indicators
    eye_status.markdown(
        f"Eyes Detected: {'✅' if detection_status['eyes_detected'] else '❌'}"
    )
    
    # Only proceed with analysis if eyes are detected
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
                
                # Store generated PDF in session state
                pdf_data = modify_generate_report(prediction, confidence, image)
                
                # Update report state
                st.session_state.report_state.pdf_bytes = pdf_data
                st.session_state.report_state.report_id = f"RPT-{uuid.uuid4().hex[:8].upper()}"
                st.session_state.report_state.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
    else:
        st.warning("Please ensure both eyes are visible in the frame.")
        quality_status.markdown("Image Quality: ❌")
        # Reset report state
        st.session_state.report_state = ReportState()

if __name__ == "__main__":
    main()
