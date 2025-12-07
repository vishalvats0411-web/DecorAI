<div align="center">
  <h1 align="center">🎨 DECORAI 🎨</h1>
  <h3><i>Where Technology Meets Design</i></h3>
  <p>
    An intelligent interior design application that uses the power of AI to help you visualize new wall colors and textures in your own space in seconds.
  </p>
</div>

<br>

![Main Showcase Image](media/demo_suggestions.png)


![Main Showcase Image](media/demo_about_me.png)
---

## 🌟 Overview
DecorAI is an intelligent web application that empowers users to visualize interior design ideas effortlessly. By leveraging the power of Computer Vision and the Segment Anything Model (SAM), this tool allows users to upload photos of their rooms, intelligently segment specific areas (like walls or furniture), and apply new colors or textures in real-time.

---

## ✨ Key Features

**Smart Segmentation**: Uses Meta's Segment Anything Model (SAM) to accurately identify and segment boundaries based on user clicks.

**Interactive Design**: Upload a room image and click on specific areas (walls, ceilings, etc.) to target them for redesign.

**Real-time Recoloring**: Apply solid colors from a preset palette or a custom color picker to the segmented areas.

**Texture Application**: Visualize how different textures (e.g., wallpapers, patterns) look on your walls.

**AI Suggestions**: Generate curated design themes and texture combinations automatically.

**User-Friendly Interface**: Built with Flask and Bootstrap for a responsive and intuitive experience.

---

## 🛠️ Tech Stack

**Backend**: Python, Flask

**AI/ML**: PyTorch, Segment Anything Model (SAM), OpenCV (cv2), NumPy

**Image Processing**: Pillow (PIL)

**Frontend**: HTML5, CSS3, Bootstrap 4, JavaScript (jQuery)

---

### Project Structure

```
  DecorAI/
├── app.py                  # Main Flask application entry point
├── requirements.txt        # Python dependencies
├── checkpoints/            # Folder for AI model weights (create manually)
├── static/
│   ├── css/                # Stylesheets
│   ├── images/             # Static assets (favicons, profile pics)
│   ├── textures/           # Texture patterns for wall application
│   ├── uploads/            # User uploaded images
│   ├── results/            # Generated segmentation results
│   └── suggestions/        # AI-generated design suggestions
└── templates/
    ├── home.html           # Landing page & upload interface
    ├── decor.html          # Main editor interface
    └── suggestions.html    # Generated themes gallery
```

### ⚙️ Installation & Setup
Follow these steps to get the project running on your local machine.

Clone the project -
```
git clone https://github.com/vishalvats0411-web/decorai.git
cd decorai
```
Download the SAM model from the [link](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth)
``` 
 create a folder named checkpoints inside main/static directory and paste the model file in there
```

Create a Virtual Environment (Recommended)
```
python -m venv venv
# Windows
venv\Scripts\activate
# Mac/Linux
source venv/bin/activate
```

Install Dependencies

```
pip install -r requirements.txt
```

Run the Application
```
python app.py
```
The server will start at http://127.0.0.1:5000/

###🎨 How to Use

***Upload***: On the home page, drag and drop or select a photo of your room (JPEG/PNG).

***Select Areas***: Once uploaded, click on the specific parts of the image you want to change (e.g., click on a wall). A red dot will appear marking your selection.

***Generate***: Click "Generate My Design". The AI will process the image and segment the selected area.

***Colors***: Use the dropdowns or the color picker to change the color of the segmented area.

***Suggestions***: Click "Our Suggestions" to see AI-generated themes and texture ideas.

---

<h3 align="center"><b>Developed with :heart: by <a href="https://github.com/vishalvats0411-web">Vishal Kumar Vats</a>
