# 🔒 SafeScope
SafeScope is a lightweight , surveillance web app built with python and YOLO V8. It detects , processes and manages camera inputs - images and videos and outputs them.

---

## 🚀 Features

- 📷 Real-time camera input processing
- 📁 Secure file uploads
- 📊 Web interface using Flask (HTML/CSS/JS)
- 🔊 Audio alerts (`.mp3` and `.wav`)
- 🎨 Styled frontend with custom icons
- 📂 Organized output directory for results

---
## 🗂️ Project Structure

create uploads and outputs folder

SafeScope/
│
├── static/ # CSS files and icons
│ ├── styles.css
│ └── ...
│
├── templates/ # HTML templates
│ ├── index.html
│ ├── result.html
│ └── ...
│
├── uploads/ # Uploaded files (auto-created)
│
├── outputs/ # Processed results (auto-created)
│
├── app.py # Main Flask application
├── requirements.txt # Python dependencies
├── .gitignore

---

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/safescope.git
   cd safescope

2. Create and activate a virtual environment

    python3 -m venv env
    source env/bin/activate   # On Windows: env/Scripts/activate

3. Install dependencies

    pip install -r requirements.txt

4. Run the app

    python app.py
