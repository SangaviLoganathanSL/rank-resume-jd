from flask import Flask, request, jsonify, send_file
from dotenv import load_dotenv
from flask_cors import CORS
import os
import fitz 
import docx
import pandas as pd
import json
import logging
from werkzeug.utils import secure_filename
import google.generativeai as genai
from datetime import datetime
from flask_swagger_ui import get_swaggerui_blueprint



# register the flask app
app = Flask(__name__)
CORS(app)
app.config['UPLOAD_FOLDER'] = "uploads"
# create a directory called uploads if it doesnt exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Gemini API Key (Read from environment variable for security)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Allowed file extensions
ALLOWED_EXTENSIONS = {"pdf", "docx"}

# Logging setup
os.makedirs('logs',exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    handlers=[logging.FileHandler(f'logs/app_{datetime.now().strftime("%Y%m%d")}.log')]
    )


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Swagger UI Setup
SWAGGER_URL = '/swagger'
API_URL = '/static/swagger.json'  # Define your Swagger JSON file
swaggerui_blueprint = get_swaggerui_blueprint(
    SWAGGER_URL,
    API_URL,
    config={'app_name': "Resume Ranking API"})
app.register_blueprint(swaggerui_blueprint, url_prefix=SWAGGER_URL)

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text("text") + "\n"
    return text

# Function to extract text from DOCX
def extract_text_from_docx(docx_path):

    with open(docx_path, "rb") as f:
        doc = docx.Document(f)  # Open document from file stream
        text = [para.text for para in doc.paragraphs]
    return text

# Function to extract ranking criteria using LLM
def extract_ranking_criteria_llm(text):
    prompt = f"""
    Extract key ranking criteria from the job description and format them EXACTLY as a JSON object.
    For each category, list all relevant items found.

    Job Description:
    {text}

    Format as:
    {{
        "skills": "List all required skills here",
        "certifications": "List all required certifications here",
        "experience": "List all experience requirements here",
        "technical_stack": "List all technical stack requirements here"
    }}
    """
    
    try:
        # Create a model instance
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Generate response
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.1,
                top_p=0.8,
                top_k=40,
            )
        )
        
        # Clean the response text
        response_text = response.text.strip()
        response_text = response_text.replace('```json', '').replace('```', '').strip()
        
        logging.info(f"Raw Gemini response: {response_text}")
        
        try:
            raw_response = json.loads(response_text)
            
            # Process each category and convert to lists
            processed_response = {}
            for category in ['skills', 'certifications', 'experience', 'technical_stack']:
                if category in raw_response:
                    # Get the raw text for this category
                    raw_text = raw_response[category]
                    
                    # Split the text into a list based on common delimiters
                    items = []
                    if isinstance(raw_text, str):
                        # Split on commas, semicolons, newlines, and bullet points
                        for item in raw_text.replace('•', ',').replace(';', ',').replace('\n', ',').split(','):
                            # Clean up each item
                            cleaned_item = item.strip()
                            if cleaned_item and cleaned_item not in ['and', 'or']:  # Remove common conjunctions
                                items.append(cleaned_item)
                    elif isinstance(raw_text, list):
                        items = raw_text  # Already a list
                    
                    # Remove duplicates while preserving order
                    seen = set()
                    processed_response[category] = [x for x in items if x and not (x in seen or seen.add(x))]
                else:
                    processed_response[category] = []
            
            logging.info(f"Processed response: {json.dumps(processed_response, indent=2)}")
            return processed_response
            
        except json.JSONDecodeError as e:
            logging.error(f"JSON parsing error: {str(e)}")
            logging.error(f"Problematic response: {response_text}")
            return {"error": "Failed to parse response as JSON"}
            
    except Exception as e:
        logging.error(f"Error calling Gemini API: {str(e)}")
        return {"error": f"API call failed: {str(e)}"}
    
@app.route('/extract-criteria', methods=['POST'])
def extract_criteria():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Extract text from file
        if filename.endswith(".pdf"):
            extracted_text = extract_text_from_pdf(filepath)
        elif filename.endswith(".docx"):
            extracted_text = extract_text_from_docx(filepath)
        else:
            return jsonify({"error": "Unsupported file format"}), 400
        
        # Extract key criteria using LLM
        extracted_criteria = extract_ranking_criteria_llm(extracted_text)
        
        return jsonify({"criteria": extracted_criteria}), 200
    else:
        return jsonify({"error": "Invalid file format"}), 400

def score_resume_against_criteria(resume_text, criteria):
    # Convert criteria list to a formatted string for the prompt
    criteria_text = "\n".join([f"- {c}" for c in criteria])
    
    prompt = f"""
    Score this resume against each criterion. Return EXACTLY in this JSON format, nothing else:
    {{
        "scores": {{
            "criterion_name": numeric_score,
            ...
        }}
    }}

    Resume:
    {resume_text}

    Criteria to score (0-5 scale):
    {criteria_text}

    Use this scoring scale:
    0: Not mentioned
    1: Barely mentioned
    3: Moderately relevant
    5: Highly relevant match
    """
    
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt)
        
        # Clean response text
        response_text = response.text.strip()
        response_text = response_text.replace('```json', '').replace('```', '').strip()
        logging.info(f"Raw Gemini response: {response_text}")
        
        try:
            scores_dict = json.loads(response_text)
            
            # Ensure we have a scores object
            if not isinstance(scores_dict, dict) or 'scores' not in scores_dict:
                raise ValueError("Invalid response format")
            
            # Calculate total and average
            total_score = sum(scores_dict['scores'].values())
            avg_score = total_score / len(scores_dict['scores']) if scores_dict['scores'] else 0
            
            return {
                "scores": scores_dict['scores'],
                "total_score": total_score,
                "average_score": round(avg_score, 2)
            }
            
        except (json.JSONDecodeError, ValueError) as e:
            logging.error(f"Error parsing response: {str(e)}")
            # Return default scores
            return {
                "scores": {criterion: 0 for criterion in criteria},
                "total_score": 0,
                "average_score": 0
            }
            
    except Exception as e:
        logging.error(f"API Error: {str(e)}")
        return {
            "scores": {criterion: 0 for criterion in criteria},
            "total_score": 0,
            "average_score": 0
        }

@app.route('/score-resumes', methods=['POST'])
def score_resumes():
    if 'criteria' not in request.form:
        return jsonify({"error": "Missing criteria"}), 400
        
    criteria = request.form.getlist('criteria')
    if not criteria:
        return jsonify({"error": "No criteria provided"}), 400
    
    if 'files' not in request.files:
        return jsonify({"error": "No files provided"}), 400
        
    files = request.files.getlist('files')
    if not files or files[0].filename == '':
        return jsonify({"error": "No files selected"}), 400
    
    results = []
    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            try:
                # Extract text from resume
                if filename.endswith(".pdf"):
                    resume_text = extract_text_from_pdf(filepath)
                elif filename.endswith(".docx"):
                    resume_text = extract_text_from_docx(filepath)
                else:
                    continue
                
                # Score resume against criteria
                scores = score_resume_against_criteria(resume_text, criteria)
                scores["filename"] = filename  # Add filename to results
                results.append(scores)
                del resume_text
                
            except Exception as e:
                logging.error(f"Error processing {filename}: {str(e)}")
                # Add error result
                results.append({
                    "filename": filename,
                    "scores": {criterion: 0 for criterion in criteria},
                    "total_score": 0,
                    "average_score": 0,
                    "error": str(e)
                })
                
            finally:
                # Clean up the file
                if os.path.exists(filepath):
                    os.remove(filepath)
    
    if not results:
        return jsonify({"error": "No valid results generated"}), 400
    
    try:
        # Convert results to DataFrame
        df_data = []
        for result in results:
            row = {
                'Filename': result['filename'],
                'Total Score': result['total_score'],
                'Average Score': result['average_score']
            }
            row.update(result['scores'])  # Add individual criteria scores
            df_data.append(row)
            
        df = pd.DataFrame(df_data)
        
        # Reorder columns to put filename first
        cols = ['Filename', 'Total Score', 'Average Score'] + [col for col in df.columns if col not in ['Filename', 'Total Score', 'Average Score']]
        df = df[cols]
        
        # Save to Excel
        output_filepath = os.path.join(app.config['UPLOAD_FOLDER'], "resume_scores.xlsx")
        df.to_excel(output_filepath, index=False)
        
        return send_file(
            output_filepath,
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            as_attachment=True,
            download_name='resume_scores.xlsx'
        )
        
    except Exception as e:
        logging.error(f"Error creating Excel file: {str(e)}")
        return jsonify({"error": "Failed to generate results file"}), 500
    
if __name__ == '__main__':
    app.run(debug=False)
