Anime recommendation system built with Streamlit that uses Item-Based Collaborative Filtering to suggest similar anime based on user ratings.


1. **Install dependencies**:
   ```bash
   in terminal type:
   py -m pip install -r requirements.txt
   ```

2. **Ensure data files are present**:
   Make sure these CSV files are in the same directory as `app.py`:
   - `anime.csv` (included in the repo via Git LFS)
   - `anime_with_synopsis.csv` (included in the repo via Git LFS)
   - `rating_complete.csv` (**not included**, ~800 MB)

   `rating_complete.csv` and `animelist.csv` are too large for GitHub. Download them from the
   [Anime Recommendation Database 2020](https://www.kaggle.com/datasets/hernan4444/anime-recommendation-database-2020)
   dataset on Kaggle and place them next to `app.py`. Clone with [Git LFS](https://git-lfs.com) installed to fetch the smaller CSVs.

## Usage

1. **Run the application**:
   ```bash
   terminal:
   streamlit run app.py
   ```

2. **Open your browser** and navigate to the URL shown in the terminal (usually `http://localhost:8501`)

3. **Enter an anime name** in the search box (e.g., "Naruto", "One Piece", "Death Note")

4. **Adjust the number of recommendations** using the slider (5-20 recommendations)

5. **Click "Get Recommendations"** to see similar anime

## How It Works

The recommendation system follows these steps:

1. **Data Loading**: Loads anime metadata, ratings, and synopsis data
2. **Data Preprocessing**: 
   - Aggregates ratings by anime (mean rating and vote count)
   - Filters to only include anime in the top 20% by vote count
   - Creates a user-anime rating matrix
3. **Similarity Calculation**: Uses Pearson correlation to find anime similar to the input
4. **Recommendation Generation**: Returns the most similar anime with their correlation scores

## Error Handling

The application includes comprehensive error handling for:
- Missing or corrupted data files
- Invalid anime names
- Anime with insufficient rating data
- Data processing errors
- Network and file system issues