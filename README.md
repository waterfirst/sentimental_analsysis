# Social Media Sentiment Analysis Dashboard

This project provides a comprehensive sentiment analysis dashboard for social media comments. It features automated sentiment scoring, interactive visualizations, and word cloud analysis.

## Features
- **Data Exploration**: Quick overview of the comments dataset.
- **Sentiment Analysis**: Automatic scoring using VADER (Valence Aware Dictionary and sEntiment Reasoner).
- **Interactive Charts**:
    - Sentiment distribution (Pie Chart)
    - Platform-specific comparison (Bar Chart)
    - Time-series trend analysis per platform (Line Chart)
- **Word Cloud Analysis**: Keyword visualization for Total, Positive, and Negative comments with shape masking.
- **Performance Metrics**: Average sentiment scores by platform.

## Tech Stack
- **Python**
- **Streamlit**: Web dashboard framework
- **Pandas**: Data manipulation
- **Plotly**: Interactive visualizations
- **VADER**: Sentiment analysis engine
- **WordCloud**: Keyword visualization

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/waterfirst/sentimental_analsysis.git
   cd sentimental_analsysis
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the dashboard:
   ```bash
   streamlit run sentiment_dashboard.py
   ```

## Dataset
The app expects a CSV file at `datasets/social_media_comments.csv` with the following columns:
- `comment_id`
- `platform`
- `comment`
- `timestamp`
