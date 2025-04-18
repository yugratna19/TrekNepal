# TrekNepal

A smart Trek Recommendation System that helps users discover perfect trekking routes in Nepal based on their preferences such as budget, available time, altitude preferences, and difficulty level. The application leverages a K-Nearest Neighbors (KNN) algorithm to match users with their ideal trekking experiences.

## Features

- **Personalized Recommendations**: Get trek suggestions tailored to your specific preferences
- **Multi-criteria Filtering**: Filter treks by cost, duration, maximum altitude, and difficulty level
- **Interactive Interface**: User-friendly Streamlit interface for easy preference selection
- **Detailed Trek Information**: View comprehensive information about each recommended trek
- **Data-driven Suggestions**: Recommendations powered by machine learning using real Nepal trekking data

## Screenshot

![Website Screenshot](output.png)

## Technology Stack

- **Python 3.7+**: Core programming language
- **Streamlit**: For creating the interactive web interface
- **Pandas**: For data manipulation and analysis
- **Scikit-learn**: For implementing the KNN recommendation algorithm
- **Matplotlib/Seaborn**: For data visualization (if used)

## Installation

### Prerequisites
- Python 3.7 or higher
- Git

### Setup Instructions

```bash
# Clone this repository
git clone https://github.com/yugratna19/TrekNepal

# Navigate to the project directory
cd TrekNepal

# Install dependencies
pip install -r requirements.txt

# Train the recommendation model
jupyter notebook model.ipynb

# Launch the application
streamlit run app.py
```

After running the application, open your web browser and go to `http://localhost:8501` to use TrekNepal.

## How It Works

1. The system loads a dataset of Nepal trekking routes with various features
2. Users input their preferences for trek cost, duration, altitude, and difficulty
3. The KNN algorithm identifies treks that most closely match these preferences
4. The application displays the recommended treks with detailed information

## Project Structure

```
TrekNepal/
├── app.py                 # Main Streamlit application
├── model.ipynb            # Jupyter notebook for model training
├── requirements.txt       # Project dependencies
├── data/                  # Dataset directory
│   └── nepal_treks.csv    # Trek dataset
├── models/                # Saved model files
└── README.md              # Project documentation
```

## Future Improvements

- Add user authentication and profile saving
- Implement trek reviews and ratings
- Add map visualization of trek routes
- Include seasonal recommendations
- Expand dataset with more trek details and images

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- [Nepal Treking Dataset](https://www.kaggle.com/datasets/bibekrai44/nepal-treking-dataset) for providing the trekking data
- [Streamlit](https://streamlit.io/) for the wonderful web app framework
- [Scikit-learn](https://scikit-learn.org/) for machine learning tools
