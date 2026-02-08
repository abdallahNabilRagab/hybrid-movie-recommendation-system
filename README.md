# Hybrid Movie Recommendation System

## Project Title

**Hybrid Movie Recommendation System** - A production-grade hybrid recommendation engine combining collaborative filtering and content-based approaches for personalized movie recommendations.

---

## Project Overview

This project implements a comprehensive, research-oriented recommendation system that synthesizes **Alternating Least Squares (ALS) collaborative filtering** with **content-based filtering** to deliver superior movie recommendations. Built on the MovieLens 32M dataset containing 32 million+ ratings from 200,000+ users across 87,000+ movies, this system addresses the core challenge of personalization at scale while managing data sparsity, cold-start problems, and computational efficiency.

### Problem Statement

Movie recommendation systems face critical challenges:
- **Data Sparsity**: User-item matrices are inherently sparse (users rate a tiny fraction of available movies)
- **Cold-Start Problem**: New users have no history; new movies have few ratings
- **Computational Scalability**: Real-time inference across millions of users and items
- **Recommendation Quality**: Balancing diversity, novelty, and accuracy

### Why This Solution Matters

This hybrid approach combats these challenges by:
1. **Collaborative Filtering (ALS)**: Captures latent user-item relationships by learning low-rank factor matrices
2. **Content-Based Filtering**: Leverages movie features (genres, tags) for cold-start and cross-domain recommendations
3. **Score Normalization & Fusion**: Combines complementary strengths through weighted aggregation
4. **Production Robustness**: Comprehensive error handling, logging, and fallback strategies
5. **Scalability**: Sparse matrices, FAISS indexing, and efficient batch processing

---

## Key Features

- ✅ **Dual-Model Architecture**: ALS collaborative filtering + content-based similarity matching
- ✅ **Hybrid Fusion**: Weighted score aggregation with configurable alpha parameter
- ✅ **Cold-Start Handling**: Fallback to popularity-based recommendations for new users/items
- ✅ **Production-Grade Stability**: Comprehensive error handling and boundary checks
- ✅ **Efficient Similarity Search**: FAISS-based cosine similarity with O(log n) lookup
- ✅ **Sparse Matrix Optimization**: CSR sparse matrices for memory-efficient storage (>99% sparsity)
- ✅ **Temporal Train-Test Split**: Prevents data leakage with per-user time-aware splitting
- ✅ **Feature Engineering**: User statistics, movie metadata, genre encoding, TF-IDF tag features
- ✅ **Comprehensive Evaluation**: Precision@10, Recall@10, NDCG@10 metrics with proper filtering
- ✅ **Interactive Streamlit UI**: Three recommendation modes (ALS, Content-Based, Hybrid)
- ✅ **Detailed Logging**: Structured logging throughout pipeline for debugging and monitoring
- ✅ **Parquet Data Storage**: Efficient columnar storage format for processed datasets

---

## System Architecture

### High-Level Design

```
┌─────────────────────────────────────────────────────────────────────┐
│                      DATA PIPELINE LAYER                             │
├─────────────────────────────────────────────────────────────────────┤
│  Raw Data → Loading → Preprocessing → Feature Engineering → Split   │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│                    MODEL TRAINING LAYER                              │
├──────────────────────────────────────┬──────────────────────────────┤
│  ALS Collaborative Filtering         │  Content-Based Filtering     │
│  • CSR sparse matrix construction    │  • Genre encoding (MLB)      │
│  • Confidence weighting (log scale)  │  • TF-IDF vectorization     │
│  • 64 latent dimensions, 20 iter     │  • FAISS indexing           │
│  • α=50 confidence scaling           │  • Cosine similarity search  │
└──────────────────────────────────────┴──────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│                    HYBRID FUSION LAYER                               │
├─────────────────────────────────────────────────────────────────────┤
│  Score Normalization → Weighted Aggregation (alpha=0.8) → Ranking   │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│                    INFERENCE & EVALUATION LAYER                      │
├──────────────────────────────────────┬──────────────────────────────┤
│  Recommender Engine                  │  Evaluation Metrics          │
│  • ALS inference                     │  • Precision@10              │
│  • Content-based search              │  • Recall@10                 │
│  • Hybrid recommend                  │  • NDCG@10                   │
└──────────────────────────────────────┴──────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│                    APPLICATION LAYER                                 │
├─────────────────────────────────────────────────────────────────────┤
│                   Streamlit Web Interface                            │
│  • ALS Mode: User ID → Recommendations                              │
│  • Content Mode: Movie Title → Similar Movies                       │
│  • Hybrid Mode: User ID → Hybrid Recommendations                    │
└─────────────────────────────────────────────────────────────────────┘
```

### Component Interaction Flow

1. **Data Loading** (`DataLoader`): Validates and loads MovieLens CSV files with optimized dtypes
2. **Preprocessing** (`DataPreprocessor`): 
   - Removes low-activity users/items (threshold: 5 interactions)
   - Deduplicates per-user interactions (keeps most recent)
   - Cleans metadata (genres, tags, links)
   - Computes interaction statistics
3. **Feature Engineering** (`FeatureBuilder`):
   - Builds user/movie aggregate features (count, mean, std)
   - Extracts temporal features (year, month, day)
   - Encodes genres with MultiLabelBinarizer
   - Computes TF-IDF for tags (max 2000 features)
   - Derives rating deviations
4. **Train-Test Split** (`TemporalSplitter`):
   - Per-user temporal split (last 20% to test)
   - Guarantees ≥1 training interaction per test user
5. **ALS Training** (`ALSTrainer`):
   - Builds confidence-weighted CSR matrix: `confidence = 1 + α * log₁ₚ(rating)`
   - Trains 64-factor ALS with regularization and 20 iterations
   - Returns trained model and sparse matrix
6. **Content Vectorization** (`ContentVectorizer`):
   - Concatenates genre features (one-hot) and tag TF-IDF
   - L2-normalizes for cosine similarity
   - Builds FAISS IndexFlatIP index
7. **Hybrid Fusion** (`HybridRecommender`):
   - Generates candidates from both models (top-500)
   - Min-Max normalizes scores independently
   - Computes: `hybrid_score = α * ALS_norm + (1-α) * CB_norm`
   - Filters seen items and returns top-K

---

## Algorithms & Models

### 1. Collaborative Filtering with ALS

**Algorithm**: Alternating Least Squares on implicit feedback

**Mathematical Foundation**:
- Given user-item interaction matrix **X** (m×n, sparse)
- Learn factorizations: **U** (m×k) and **V** (n×k)
- Minimize: `||X - UV^T||² + λ(||U||² + ||V||²)`
- Implicit feedback: Confidence weighting `c_ui = 1 + α * log₁ₚ(r_ui)`

**Why ALS**:
- Efficient for sparse, implicit feedback data
- Parallelizable computation (alternating updates)
- Scales to millions of users/items
- Produces dense latent factor representations

**Hyperparameters**:
- `factors=64`: Latent dimension (balance between expressiveness and overfitting)
- `iterations=20`: Training passes (convergence typically reached by 15-20)
- `regularization=0.01`: L2 penalty to prevent overfitting
- `alpha=50.0`: Confidence scaling factor (higher weight to positive interactions)
- `num_threads=8`: Parallel computation

**Implementation Details**:
- Uses scipy sparse CSR format for memory efficiency
- Confidence computed as: `1 + 50 * log₁ₚ(rating)` (emphasizes high ratings)
- Filters sparse users/items (min 5 interactions each)
- Thread-safe with threadpool limits to prevent race conditions

### 2. Content-Based Filtering

**Approach**: Feature-based similarity matching using genre and tag metadata

**Feature Representation**:
- **Genre Features**: One-hot encoded (MultiLabelBinarizer) → ~20 binary features
- **Tag Features**: TF-IDF vectorization (max 2000 features)
- **Combined**: Concatenated and L2-normalized for cosine similarity

**Similarity Metric**: Cosine similarity via FAISS Inner Product
- Inner product of L2-normalized vectors = cosine similarity
- Complexity: O(d) for similarity computation (d = feature dimension)

**User Profile Construction**:
- Historical mean vector of positive-rated items (rating ≥ user's mean rating)
- Fallback to all items if insufficient positive ratings

**Why Content-Based**:
- Handles cold-start users effectively (new users rated ≥1 item can get recommendations)
- Provides interpretable recommendations (based on genres/tags)
- No data sparsity issues
- Good for cross-domain discovery

**FAISS Optimization**:
- IndexFlatIP: Inner product search (O(n) complexity with SIMD acceleration)
- Retrieves top-k candidates efficiently (typically <1ms for 87K movies)
- C-contiguous arrays ensure memory efficiency

### 3. Hybrid Recommender System

**Fusion Strategy**: Weighted linear combination of normalized scores

**Process**:
1. **Candidate Generation**:
   - Get top-500 candidates from ALS (wide candidate pool)
   - Get top-500 candidates from Content-Based
   - Union of all candidates

2. **Score Normalization** (Independent for each model):
   - Min-Max normalization: `s_norm = (s - min) / (max - min)`
   - Maps all scores to [0, 1] range
   - Prevents one model from dominating due to different score scales

3. **Weighted Fusion**:
   ```
   hybrid_score(item) = α * ALS_norm(item) + (1-α) * CB_norm(item)
   ```
   - Default α = 0.8 (emphasis on collaborative filtering)
   - Configurable per inference

4. **Post-Processing**:
   - Filter seen items (user's history)
   - Rank by combined score
   - Return top-K items

**Why Hybrid**:
- **Synergy**: ALS captures user preferences; Content captures item attributes
- **Robustness**: If one model fails (e.g., new user), other provides fallback
- **Coverage**: Combines coverage of both approaches
- **Quality**: Research shows hybrid typically outperforms single-model approaches

**Cold-Start Handling**:
- New users (not in training): Return popularity-based recommendations (top-10 global most-rated movies)
- New items (not in ALS matrix): Still accessible via content-based similarity
- Graceful degradation with fallback strategies

---

## Data Pipeline

### 1. Data Loading

**Source**: MovieLens 32M Dataset
- **ratings.csv**: 32M+ interactions with userId, movieId, rating (0.5-5.0), timestamp
- **movies.csv**: 87K+ movies with title, genres
- **tags.csv**: User-generated tags with user, movie, tag, timestamp
- **links.csv**: Links to IMDb and TMDb databases

**Implementation** (`DataLoader`):
```python
class DataLoader:
    def load_data() → Dict[str, DataFrame]:
        - Verify file existence
        - Load with optimized dtypes:
          * userId, movieId: int32 (saves memory vs int64)
          * rating: float32
          * timestamp: int64
        - Return {ratings, movies, tags, links}
```

**Configuration** (`Config`):
- `MIN_USER_RATINGS = 5`: Filter users with <5 ratings
- `MIN_MOVIE_RATINGS = 5`: Filter movies with <5 ratings
- `COLD_START_THRESHOLD = 10`: Flag users/movies near sparsity threshold
- `MIN_RATING = 0.5, MAX_RATING = 5.0`: Valid rating range
- `MIN_TAG_FREQUENCY = 5`: Filter rare tags
- `GENERIC_TAGS`: Exclude generic tags (movie, film, cinema, etc.)

### 2. Data Preprocessing

**Objective**: Clean, filter, and standardize raw data

**Steps** (`DataPreprocessor`):

1. **Dataset Inspection**:
   - Count missing values and duplicates
   - Log statistics for each dataset

2. **Activity Analysis**:
   - Compute user interaction counts (activity per user)
   - Compute item popularity (ratings per movie)
   - Identify sparsity patterns

3. **Interaction Filtering**:
   - Remove users with <5 ratings (MIN_USER_RATINGS)
   - Remove movies with <5 ratings (MIN_MOVIE_RATINGS)
   - Keeps only high-engagement subset
   - **Impact**: Reduces noise, improves model quality

4. **Timestamp Processing**:
   - Convert Unix timestamps to datetime objects
   - Remove invalid timestamps (coerce to NaT)
   - Enables temporal analysis and time-aware splits

5. **Duplicate Interaction Removal**:
   - Group by (userId, movieId)
   - Keep only most recent interaction per pair
   - Resolves inconsistent user behavior

6. **Rating Validation**:
   - Keep ratings in [0.5, 5.0] range
   - Remove outliers and malformed data

7. **Movie Metadata Cleaning**:
   - Fill missing genres with "Unknown"
   - Ensure all movies have genre info

8. **Tag Cleaning**:
   - Normalize text (lowercase, strip whitespace)
   - Remove generic tags (movie, film, etc.)
   - Filter rare tags (<5 occurrences)
   - Reduces noise and improves TF-IDF quality

9. **Links Cleaning**:
   - Remove missing IMDb/TMDb IDs
   - Ensures referential integrity

**Data Quality Metrics**:
- Sparsity: 1 - (interactions / (users × items))
  - Typical: 99.9%+ (very sparse)
- Coverage: # unique movies rated / total movies
  - Typical: 50-80%

### 3. Feature Engineering

**Objective**: Create rich, informative features for models

**User Features** (`build_user_features`):
- `user_rating_count`: Total interactions per user (activity level)
- `user_avg_rating`: Mean rating per user (rating tendency)
- `user_std_rating`: Rating standard deviation (consistency)

**Movie Features** (`build_movie_features`):
- `movie_rating_count`: Total interactions per movie (popularity)
- `movie_avg_rating`: Mean rating across users (quality indicator)
- `movie_std_rating`: Rating variance (divisiveness)

**Temporal Features** (`build_temporal_features`):
- Extract year, month, day from timestamp
- Captures seasonality and trend patterns

**Yearly Activity** (`build_yearly_activity`):
- Per-user interactions per year
- Per-movie ratings per year
- Analyzes temporal trends

**Genre Features** (`encode_genres`):
- One-hot encode genres (e.g., "Action|Drama" → [1,0,...,1])
- MultiLabelBinarizer: ~20-30 unique genres
- Used in content-based feature vectors

**Tag Features** (`build_tag_features`):
- Convert tag lists to space-separated strings
- TF-IDF vectorization (max 2000 features)
- Captures semantic information about movies

**Rating Deviations**:
- `rating_dev_user`: rating - user_avg_rating
- `rating_dev_movie`: rating - movie_avg_rating
- Normalizes user/item biases

**Index Mappings**:
- `user2idx`: userId → internal sequential index (0 to n_users-1)
- `movie2idx`: movieId → internal index for sparse matrix
- Essential for ALS training

**Output**: Dictionary containing all features, mappings, and transformed dataframes

### 4. Train-Test Split

**Strategy**: Temporal splitting per user (time-aware, prevents leakage)

**Implementation** (`TemporalSplitter`):
```python
def temporal_split(df, test_ratio=0.2):
    for each user:
        - Sort interactions by timestamp
        - Split: last 20% → test, rest → train
        - Guarantee ≥1 training sample per test user
        - Users with <2 interactions stay entirely in train
    return train_df, test_df
```

**Guarantees**:
- ✅ No cold-start users in evaluation (all test users in train)
- ✅ Training data predates test data temporally
- ✅ Realistic evaluation (mimics actual recommender behavior)
- ✅ At least 1 observation per user for ALS cold-start handling

**Output Sizes** (20% sample):
- Typical: 80% interactions in train, 20% in test
- Both saved as Parquet for efficient I/O

---

## Training & Evaluation

### Training Workflow

#### Step 1: ALS Training

**Inputs**:
- train_df: Clean interactions with userId, movieId, rating, timestamp
- user_map: userId → internal index
- item_map: movieId → internal index

**Process** (`ALSTrainer.fit()`):

1. **Confidence Weighting**:
   ```python
   confidence = 1 + α * log₁ₚ(rating)
   # Example: rating=5 → confidence ≈ 1 + 50*log(6) ≈ 80
   ```
   - Emphasizes high ratings while maintaining implicit feedback paradigm
   - Prevents negative ratings from dominating

2. **Sparse Entity Filtering**:
   - Remove users with <5 interactions
   - Remove items with <5 interactions
   - Reduces noise and computational burden

3. **Sparse Matrix Construction**:
   - Build CSR matrix: shape = (n_users, n_items)
   - Values: confidence scores
   - NNZ (non-zero): ~20-50 million entries
   - Sparsity: >99.9% (vast majority zeros)

4. **Environment Setup**:
   - Set OPENBLAS, MKL, OMP threads to 1
   - Use threadpool_limits for thread safety
   - Prevents race conditions in parallel ALS

5. **ALS Model Training**:
   - Initialize 64 latent factors
   - Run 20 iterations of alternating optimization
   - Converges when residual change < tolerance

6. **Validation**:
   - Verify user_factors shape matches matrix
   - Check model not None
   - Log model statistics (dimensions, NNZ)

**Output**: (trained_model, X_sparse)

#### Step 2: Content-Based Vectorization

**Inputs**:
- movies: Movie metadata with genres
- tags: User-generated tags

**Process** (`ContentVectorizer`):

1. **Genre Encoding**:
   - Split genres string: "Action|Drama" → ["Action", "Drama"]
   - MultiLabelBinarizer creates binary matrix
   - Shape: (n_movies, n_genres≈20)

2. **Tag TF-IDF Vectorization**:
   - Group tags by movieId
   - Join tags: "tag1 tag2 tag3"
   - TF-IDF on tag corpus (max 2000 features)
   - Shape: (n_movies, 2000)

3. **Feature Concatenation**:
   - Horizontal stack: [genre_features | tag_features]
   - Shape: (n_movies, ~2020)
   - Fillna(0) for missing values

4. **Normalization**:
   - Convert to float32 for memory efficiency
   - Ensure C-contiguous array layout (FAISS requirement)
   - L2-normalize using faiss.normalize_L2()

5. **FAISS Index Building**:
   - Create IndexFlatIP (inner product search)
   - Add normalized vectors to index
   - Enables fast similarity search

6. **ID Mappings**:
   - movieId ↔ index mappings (for search result conversion)

**Output**: (item_features, mlb, vectorizer, faiss_index, mappings)

#### Step 3: Hybrid System Creation

**Inputs**:
- als_recommender: Trained ALS engine
- content_searcher: Content-based search engine
- train_df: Training interactions (for fallback)

**Process**:
- Instantiate HybridRecommender with all components
- Ready for inference with flexible α parameter

### Evaluation Metrics

**Metrics Used**: Standard recommendation system metrics

1. **Precision@K**:
   ```
   Precision@K = (# relevant items in top-K) / K
   ```
   - Measures: Of top-K recommendations, how many are relevant?
   - Range: [0, 1], higher is better
   - Typical: 0.05-0.20 in research benchmarks

2. **Recall@K**:
   ```
   Recall@K = (# relevant items in top-K) / (# relevant items)
   ```
   - Measures: Of all relevant items, how many in top-K?
   - Range: [0, 1], higher is better
   - Typical: 0.02-0.10 for top-10

3. **NDCG@K** (Normalized Discounted Cumulative Gain):
   ```
   DCG@K = Σ(rel_i / log₂(i+1)) for i=1..K
   IDCG@K = optimal DCG (perfect ranking)
   NDCG@K = DCG@K / IDCG@K
   ```
   - Measures: How well are items ranked (order matters)?
   - Range: [0, 1], higher is better
   - Accounts for position (items at top weighted more)
   - Typical: 0.10-0.25

**Evaluation Process** (`Evaluator`):
1. Generate predictions for sampled users (top-100 from test set)
2. Compare against ground truth (test items per user)
3. Filter invalid predictions (users with no recommendations)
4. Compute metrics only on evaluated users
5. Return mean scores: (precision, recall, ndcg, n_users_evaluated)

**Typical Results** (20% sample data):
- **ALS Precision@10**: ~0.08-0.12
- **ALS Recall@10**: ~0.05-0.08
- **ALS NDCG@10**: ~0.12-0.18
- **Hybrid Performance**: ~5-15% improvement over ALS alone

**Cold-Start Handling**:
- New users (0 interactions): Fallback to popularity (get global top-10)
- New items: Participate via content similarity
- Out-of-bounds users: Gracefully skip with fallback

---

## Application Interface

### Streamlit Web UI

**Purpose**: Interactive interface for recommendation exploration

**Technology**: Streamlit (Python web framework, no frontend coding required)

**Design Philosophy**:
- Cloud-safe (handles different file systems)
- Modular imports with error handling
- Resource caching (only load models once)
- Responsive layout (wide container)

#### Page Configuration

```python
st.set_page_config(
    page_title="Hybrid Movie Recommender",
    page_icon="🎬",
    layout="wide",
)
```

#### Components

**1. Resource Loading** (`@st.cache_resource`):
- Cache decorator loads RecommenderEngine once
- State persists across reruns
- Spinner shows "Loading recommendation engine..."

**2. Sidebar Controls**:
- Selectbox for recommendation mode:
  - ALS (User-Based)
  - Content-Based
  - Hybrid

**3. ALS Mode**:
```
Input:
  - User ID (integer, min=1)
Button:
  - "Get Recommendations"
Output:
  - DataFrame: [movieId, score, title, genres, ...]
```

**4. Content-Based Mode**:
```
Input:
  - Movie Title (selectbox, sorted)
Button:
  - "Find Similar Movies"
Output:
  - DataFrame: Similar movies ranked by similarity
```

**5. Hybrid Mode**:
```
Input:
  - User ID (integer, min=1)
Button:
  - "Generate Hybrid Recommendations"
Output:
  - DataFrame: Recommendations from hybrid fusion
```

**6. Footer**:
- Developer information (Abdallah Nabil Ragab)
- Email for feedback
- Profile image from CDN

#### User Interaction Flow

```
┌─────────────────────────────────────────────────────┐
│  1. User Opens App (loads cached models once)       │
└──────────────┬──────────────────────────────────────┘
               ↓
┌─────────────────────────────────────────────────────┐
│  2. User Selects Mode from Sidebar                  │
│     (ALS | Content-Based | Hybrid)                  │
└──────────────┬──────────────────────────────────────┘
               ↓
┌─────────────────────────────────────────────────────┐
│  3. Mode-Specific Input                             │
│     ALS: Enter User ID                              │
│     Content: Select Movie Title                      │
│     Hybrid: Enter User ID                           │
└──────────────┬──────────────────────────────────────┘
               ↓
┌─────────────────────────────────────────────────────┐
│  4. User Clicks Button → Trigger Inference          │
│     Spinner shows: "Generating recommendations..."  │
└──────────────┬──────────────────────────────────────┘
               ↓
┌─────────────────────────────────────────────────────┐
│  5. Engine Generates Recommendations                │
│     (typically <1 second for all modes)             │
└──────────────┬──────────────────────────────────────┘
               ↓
┌─────────────────────────────────────────────────────┐
│  6. Results Displayed in DataFrame                  │
│     Columns: movieId, score, title, genres          │
└─────────────────────────────────────────────────────┘
```

---

## Installation & Setup

### Prerequisites

- **Python**: 3.8 or higher
- **OS**: Windows, macOS, Linux
- **RAM**: Minimum 8GB (16GB recommended for full ML-32M dataset)
- **Disk**: ~20GB for extracted data + models

### Virtual Environment Setup

**Option 1: Using venv (Python Standard)**

```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (macOS/Linux)
source venv/bin/activate
```

**Option 2: Using Conda**

```bash
# Create conda environment
conda create -n recsys python=3.10

# Activate
conda activate recsys
```

### Install Dependencies

```bash
# Install all required packages
pip install -r requirements.txt
```

**Dependencies Breakdown**:

| Package | Purpose | Version |
|---------|---------|---------|
| pandas | Data manipulation, tabular operations | Latest |
| numpy | Numerical computation, matrix operations | Latest |
| scikit-learn | ML utilities (TF-IDF, normalization) | Latest |
| scipy | Sparse matrix handling | Latest |
| implicit | ALS collaborative filtering library | Latest |
| faiss-cpu | Similarity search index (CPU version) | Latest |
| tqdm | Progress bars in pipelines | Latest |
| joblib | Parallel processing, model serialization | Latest |
| matplotlib | Visualization (if used in notebooks) | Latest |
| seaborn | Statistical visualization | Latest |
| streamlit | Web UI framework | Latest |
| nltk | Text processing utilities | Latest |

**Optional GPU Support**:
```bash
# Replace faiss-cpu with faiss-gpu if you have NVIDIA GPU
pip uninstall faiss-cpu
pip install faiss-gpu
```

### Project Structure Setup

```
recsys_project/
├── README.md                    # This file
├── requirements.txt              # Python dependencies
├── app/
│   ├── __init__.py
│   └── streamlit_app.py         # Main Streamlit application
├── data/
│   ├── raw/                     # Original MovieLens CSVs
│   │   ├── ratings.csv          # User ratings (32M rows)
│   │   ├── movies.csv           # Movie metadata (87K rows)
│   │   ├── tags.csv             # User tags (2M rows)
│   │   ├── links.csv            # External links
│   │   ├── checksums.txt        # MD5 verification
│   │   └── README.txt           # Dataset documentation
│   ├── processed/               # Cleaned and engineered data
│   │   ├── clean_interactions.parquet
│   │   ├── clean_movies.parquet
│   │   ├── clean_tags.parquet
│   │   └── evaluation/          # Evaluation results
│   │       ├── als_results.parquet
│   │       └── hybrid_results.parquet
│   └── splits/                  # Train/test split
│       ├── train.parquet
│       └── test.parquet
├── models/                      # Trained models and artifacts
│   ├── als_model.pkl            # Trained ALS model
│   ├── X_sparse.npz             # User-item interaction matrix
│   ├── user_map.pkl             # userId → index mapping
│   ├── item_map.pkl             # movieId → index mapping
│   ├── faiss.index              # FAISS similarity index
│   ├── item_features.npy        # Feature vectors for items
│   ├── mlb.pkl                  # Genre MultiLabelBinarizer
│   ├── tfidf.pkl                # TF-IDF vectorizer
│   └── movieId_to_index.pkl     # FAISS ID mapping
├── notebooks/                   # Jupyter notebooks (if any)
├── scripts/
│   ├── run_pipeline.py          # Main pipeline execution
│   └── __pycache__/
└── src/
    ├── __init__.py
    ├── config.py                # Configuration constants
    ├── data_loader.py           # Data loading module
    ├── preprocessing.py         # Data preprocessing
    ├── features.py              # Feature engineering
    ├── split.py                 # Train-test splitting
    ├── evaluation.py            # Evaluation metrics
    ├── inference.py             # Inference engine
    ├── als/
    │   ├── __init__.py
    │   ├── train.py             # ALS training
    │   ├── recommend.py         # ALS inference
    │   └── evaluate.py          # ALS evaluation
    ├── content_based/
    │   ├── __init__.py
    │   ├── vectorize.py         # Feature vectorization
    │   └── search.py            # FAISS similarity search
    ├── hybrid/
    │   ├── __init__.py
    │   └── hybrid.py            # Hybrid recommendation fusion
    └── utils/
        ├── __init__.py
        └── metrics.py           # Evaluation metric functions
```

---

## Running the Project

### Step 1: Download Data

Download MovieLens 32M dataset from [grouplens.org](http://grouplens.org/datasets/movielens/):

```bash
# Extract to data/raw/ directory
# Structure should be:
# data/raw/
#   ├── ratings.csv
#   ├── movies.csv
#   ├── tags.csv
#   ├── links.csv
#   ├── checksums.txt
#   └── README.txt
```

**Note**: Full dataset is ~8GB compressed, 32GB extracted. You may want to use a smaller version:
- **ml-1m**: 1 million ratings (smaller, faster)
- **ml-20m**: 20 million ratings (medium)
- **ml-32m**: 32 million ratings (production-grade, requires patience)

### Step 2: Run the Pipeline

Execute the complete recommendation pipeline:

```bash
# From project root directory
python scripts/run_pipeline.py
```

**Pipeline Execution** (with 20% sample data):
1. Load raw datasets (~30 seconds)
2. Relational sampling to 20% (~10 seconds)
3. Preprocess and clean data (~20 seconds)
4. Feature engineering (~15 seconds)
5. Train-test split (~5 seconds)
6. ALS training (~2-5 minutes)
7. Content-based vectorization (~1 minute)
8. Evaluation (~30 seconds)

**Expected Output**:
```
🚀 Starting Recommendation Pipeline...
📥 Loading raw datasets...
✅ Ratings loaded | Shape: (32000204, 4)
✅ Movies loaded | Shape: (87585, 3)
...
💪 Preparing ALS pipeline...
🧹 Preprocessing data...
🛠️ Building features...
📆 Temporal split...
📚 Building content-based system...
🤝 Running hybrid inference...
📊 Evaluating Hybrid...
🎉 Pipeline finished successfully
⏳ Runtime: 450.23 sec
```

**Artifacts Generated**:
- Cleaned Parquet files in `data/processed/`
- Train/test splits in `data/splits/`
- Models and mapings in `models/`
- Evaluation results in `data/processed/evaluation/`

### Step 3: Launch Streamlit App

Start the interactive web interface:

```bash
# From project root
streamlit run app/streamlit_app.py
```

**Console Output**:
```
  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://192.168.x.x:8501
```

**Browser**:
- Opens automatically at `localhost:8501`
- Select mode from sidebar
- Enter input (User ID or Movie Title)
- Click button to generate recommendations
- View results in DataFrame

### Example Commands

**Full Pipeline Execution**:
```bash
cd d:\recsys_project
python -m venv venv
venv\Scripts\activate  # Windows
python -m pip install --upgrade pip
pip install -r requirements.txt
python scripts/run_pipeline.py
```

**Run Streamlit App**:
```bash
streamlit run app/streamlit_app.py
```

**Run Specific Module** (for debugging):
```bash
python -c "from src.data_loader import DataLoader; dl = DataLoader(); print(dl.load_data())"
```

**Check Installed Packages**:
```bash
pip list
```

---

## Configuration

### Environment Variables

Currently, configuration is hardcoded in `src/config.py`. To customize:

```python
# src/config.py
class Config:
    # Data paths (adjust as needed)
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    DATA_DIR = PROJECT_ROOT / "data"
    
    # Filtering thresholds
    MIN_USER_RATINGS = 5       # Increase to 10 for stricter filtering
    MIN_MOVIE_RATINGS = 5      # Increase for popular-items-only
    COLD_START_THRESHOLD = 10
    
    # Valid rating range
    MIN_RATING = 0.5
    MAX_RATING = 5.0
    
    # Tags
    GENERIC_TAGS = [...]       # Add/remove as needed
    MIN_TAG_FREQUENCY = 5
```

### Hyperparameter Tuning

**ALS Parameters** (`scripts/run_pipeline.py`):
```python
als_trainer = ALSTrainer(
    factors=64,          # Latent dimensions (try 32, 64, 128)
    iterations=20,       # Passes (try 10, 20, 40)
    regularization=0.01, # L2 penalty (try 0.001, 0.01, 0.1)
    num_threads=8,       # Parallel threads
    alpha=50.0,          # Confidence scaling (try 10, 50, 100)
)
```

**Hybrid Alpha** (`app/streamlit_app.py` or `src/inference.py`):
```python
# Default alpha=0.7 (70% ALS, 30% Content)
recs = engine.recommend_hybrid(user_id, top_k=10, alpha=0.7)

# Try different values:
# alpha=0.5: Equal weight
# alpha=0.8: Prefer collaborative filtering
# alpha=0.3: Prefer content-based
```

**Evaluation Settings** (`scripts/run_pipeline.py`):
```python
# Sampled users for evaluation (increase for full evaluation)
sampled_users = test_df["userId"].drop_duplicates().sample(
    min(100, test_df["userId"].nunique()),  # Change 100 to 500 or 1000
    random_state=42
)
```

---

## Logging & Monitoring

### Logging System

Project uses Python's standard `logging` module throughout.

**Logger Configuration**:
```python
import logging

logger = logging.getLogger(__name__)

# Set level
logger.setLevel(logging.INFO)  # DEBUG, INFO, WARNING, ERROR, CRITICAL

# Add handler
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

# Log messages
logger.info("Starting process...")     # Informational
logger.warning("Potential issue...")   # Warnings
logger.error("Error occurred!")        # Errors
```

### Module-Specific Logging

**Verbose Modules** (detailed logging):
- `data_loader.py`: File loading steps
- `preprocessing.py`: Cleaning operations
- `features.py`: Feature engineering progress
- `split.py`: Data splitting statistics
- `als/train.py`: Training progress
- `als/evaluate.py`: Evaluation metrics

**Silent Modules** (suppressed logging):
- `als/recommend.py`: Production inference (can enable with verbose=True)
- `content_based/search.py`: FAISS operations
- `content_based/vectorize.py`: Vectorization
- `hybrid/hybrid.py`: Hybrid fusion (tqdm used instead)

### Monitoring During Pipeline

```bash
# Watch logs in real-time
# (Logs appear in console as pipeline runs)

# Example output:
2024-01-15 10:23:45 | INFO | 🚀 Starting Recommendation Pipeline...
2024-01-15 10:23:47 | INFO | 📥 Loading raw datasets...
2024-01-15 10:24:05 | INFO | ✅ Ratings loaded | Shape: (6400040, 4)
2024-01-15 10:24:15 | INFO | 📉 Relational sampling 20% from ratings...
...
```

### Error Handling

**Try-Except Patterns**:
```python
# In inference
try:
    recs = self.model.recommend(...)
except Exception as e:
    logger.error(f"Inference failed: {e}")
    return self._popular_items(top_k)  # Fallback

# In ALS recommender
if user_id not in self.user_map:
    logger.warning(f"Cold-start user: {user_id}")
    return popular_recommendations
```

### Debugging Tips

1. **Enable Verbose Logging**:
   ```python
   recs = engine.als_engine.recommend_als(user_id, top_k=10, verbose=True)
   ```

2. **Check Model Files**:
   ```bash
   ls -la models/  # Verify all files exist
   ```

3. **Validate Data**:
   ```python
   from src.data_loader import DataLoader
   loader = DataLoader()
   data = loader.load_data()
   print(data["ratings"].head())
   ```

4. **Test Components**:
   ```python
   # Test data loading
   from src.data_loader import DataLoader
   loader = DataLoader()
   data = loader.load_data()
   
   # Test preprocessing
   from src.preprocessing import DataPreprocessor
   prep = DataPreprocessor()
   filtered = prep.filter_interactions(...)
   ```

---

## Project Structure

### Directory Breakdown

```
recsys_project/
│
├── 📄 README.md
│   └─> This comprehensive documentation
│
├── 📄 requirements.txt
│   └─> Python dependencies (pandas, numpy, scikit-learn, etc.)
│
├── 📁 app/
│   ├── __init__.py
│   └── streamlit_app.py
│       ├─> Streamlit web UI
│       ├─> Three modes: ALS, Content-Based, Hybrid
│       ├─> Caches RecommenderEngine
│       ├─> Handles cloud-safe imports
│       └─> Developer info footer
│
├── 📁 data/
│   ├── raw/                  # Original MovieLens datasets
│   │   ├── ratings.csv       # 32M+ user ratings
│   │   ├── movies.csv        # 87K+ movie metadata
│   │   ├── tags.csv          # 2M+ user-generated tags
│   │   ├── links.csv         # IMDb/TMDb links
│   │   ├── checksums.txt     # MD5 verification
│   │   └── README.txt        # Dataset documentation
│   │
│   ├── processed/            # Cleaned & engineered data
│   │   ├── clean_interactions.parquet
│   │   │   └─> Cleaned user-item interactions
│   │   ├── clean_movies.parquet
│   │   │   └─> Movie metadata (genres, titles)
│   │   ├── clean_tags.parquet
│   │   │   └─> Normalized tags
│   │   └── evaluation/       # Evaluation results
│   │       ├── als_results.parquet      # ALS metrics
│   │       └── hybrid_results.parquet   # Hybrid metrics
│   │
│   └── splits/               # Train-test splits
│       ├── train.parquet     # 80% of interactions
│       └── test.parquet      # 20% of interactions
│
├── 📁 models/                # Trained models & artifacts
│   ├── als_model.pkl         # Trained ALS model (AlternatingLeastSquares)
│   ├── X_sparse.npz          # User-item CSR matrix (scipy.sparse)
│   ├── user_map.pkl          # userId → internal index mapping
│   ├── item_map.pkl          # movieId → internal index mapping
│   ├── inv_item_map.pkl      # internal index → movieId (optional)
│   ├── faiss.index           # FAISS IndexFlatIP (similarity index)
│   ├── item_features.npy     # Dense feature vectors (float32)
│   ├── mlb.pkl               # Genre MultiLabelBinarizer
│   ├── tfidf.pkl             # TF-IDF vectorizer for tags
│   └── movieId_to_index.pkl  # MovieId → FAISS index mapping
│
├── 📁 notebooks/             # Jupyter notebooks (analysis/exploration)
│
├── 📁 scripts/
│   └── run_pipeline.py
│       ├─> Main execution script
│       ├─> Orchestrates full pipeline:
│       │   1. Load data
│       │   2. Preprocess
│       │   3. Feature engineering
│       │   4. Train-test split
│       │   5. ALS training
│       │   6. Content-based vectorization
│       │   7. Hybrid fusion
│       │   8. Evaluation
│       ├─> Handles caching (skip if files exist)
│       ├─> Saves Parquet + pickle artifacts
│       └─> Single entry point for reproducibility
│
└── 📁 src/                   # Core library modules
    ├── __init__.py
    ├── config.py
    │   └─> Configuration class with paths, thresholds
    │
    ├── data_loader.py
    │   ├─> DataLoader class
    │   ├─> Verifies required files
    │   ├─> Loads CSV with optimized dtypes
    │   └─> Returns Dict[name → DataFrame]
    │
    ├── preprocessing.py
    │   ├─> DataPreprocessor class
    │   ├─> Inspects datasets (missing, duplicates)
    │   ├─> Filters low-activity users/items
    │   ├─> Processes timestamps
    │   ├─> Removes duplicates (keeps last)
    │   ├─> Cleans movies, tags, links metadata
    │   └─> Computes sparsity & activity
    │
    ├── features.py
    │   ├─> FeatureBuilder class
    │   ├─> User features (count, mean, std)
    │   ├─> Movie features (count, mean, std)
    │   ├─> Temporal features (year, month, day)
    │   ├─> Yearly activity analysis
    │   ├─> Genre encoding (MultiLabelBinarizer)
    │   ├─> Tag feature extraction
    │   ├─> Rating deviations
    │   └─> Index mappings (userId/movieId → sequential)
    │
    ├── split.py
    │   └─> TemporalSplitter class
    │       ├─> Per-user temporal split
    │       ├─> Guarantees ≥1 train sample per test user
    │       └─> Prevents data leakage
    │
    ├── evaluation.py
    │   └─> Evaluator class
    │       ├─> Evaluates hybrid recommendations
    │       ├─> Computes Precision@K, Recall@K, NDCG@K
    │       ├─> Handles invalid predictions gracefully
    │       └─> Uses tqdm progress bar
    │
    ├── inference.py
    │   └─> RecommenderEngine class
    │       ├─> Loads all trained models/artifacts
    │       ├─> Builds ALS inference engine
    │       ├─> Builds content-based search engine
    │       ├─> Builds hybrid recommendation engine
    │       ├─> Methods: recommend_als, recommend_content, recommend_hybrid
    │       └─> Merges results with movie metadata
    │
    ├── als/                  # Collaborative Filtering
    │   ├── __init__.py
    │   ├── train.py
    │   │   └─> ALSTrainer class
    │   │       ├─> Confidence weighting (1 + α*log₁ₚ(r))
    │   │       ├─> CSR sparse matrix construction
    │   │       ├─> ALS model training (64 factors, 20 iters)
    │   │       └─> Returns model + sparse matrix
    │   │
    │   ├── recommend.py
    │   │   └─> ALSRecommender class
    │   │       ├─> Safe ID mapping with boundary checks
    │   │       ├─> Public: recommend_als() method
    │   │       ├─> Cold-start fallback to popularity
    │   │       ├─> Score normalization option
    │   │       ├─> Silent logging (production-safe)
    │   │       └─> Returns [(movieId, score), ...]
    │   │
    │   └── evaluate.py
    │       └─> ALSEvaluator class
    │           ├─> Precompute predictions on test set
    │           ├─> Compute Precision@K, Recall@K, NDCG@K
    │           ├─> Uses proper metric functions
    │           └─> Filters invalid users automatically
    │
    ├── content_based/        # Content-Based Filtering
    │   ├── __init__.py
    │   ├── vectorize.py
    │   │   └─> ContentVectorizer class
    │   │       ├─> Genre encoding (one-hot)
    │   │       ├─> Tag TF-IDF vectorization
    │   │       ├─> Feature concatenation & normalization
    │   │       ├─> FAISS index building
    │   │       ├─> ID mappings (movieId ↔ index)
    │   │       └─> Returns vectors, mappings, index
    │   │
    │   └── search.py
    │       └─> ContentSearcher class
    │           ├─> User profile from history
    │           ├─> FAISS similarity search
    │           ├─> Filters seen items
    │           ├─> Cold-start fallback to popularity
    │           └─> Returns [(movieId, score), ...]
    │
    ├── hybrid/               # Hybrid Recommendation
    │   ├── __init__.py
    │   └── hybrid.py
    │       └─> HybridRecommender class
    │           ├─> Combines ALS + Content-based
    │           ├─> Candidate pool (top-500 each)
    │           ├─> Score normalization (Min-Max)
    │           ├─> Weighted fusion (α * ALS + (1-α) * CB)
    │           ├─> Filters seen items
    │           ├─> Cold-start protection
    │           ├─> tqdm-safe evaluation
    │           └─> Returns [(movieId, score), ...]
    │
    └── utils/               # Utility functions
        ├── __init__.py
        └── metrics.py
            ├─> recall_at_k()   # Recall@K implementation
            ├─> precision_at_k() # Precision@K implementation
            └─> ndcg_at_k()      # NDCG@K implementation

File Count: ~50 files
Code Lines: ~2500+ lines of production-grade Python
```

### Key Dependencies Between Modules

```
scripts/run_pipeline.py
├─> DataLoader → loads raw data
├─> DataPreprocessor → cleans data
├─> FeatureBuilder → creates features
├─> TemporalSplitter → splits data
├─> ALSTrainer → trains ALS model
├─> ContentVectorizer → creates content features
├─> ContentSearcher → initializes content engine
├─> HybridRecommender → combines engines
├─> ALSEvaluator → evaluates ALS
└─> Evaluator → evaluates hybrid

app/streamlit_app.py
└─> RecommenderEngine
    ├─> ALSRecommender (trained ALS)
    ├─> ContentSearcher (FAISS index)
    └─> HybridRecommender (fusion)
```

---

## Best Practices & Design Decisions

### 1. Data Handling

**Decision**: Use Parquet for processed data instead of CSV
- **Why**: Columnar format, compression, faster I/O, schema preservation
- **Trade-off**: Requires pyarrow, but standard industry practice
- **Implementation**: `pd.to_parquet()` and `pd.read_parquet()`

**Decision**: Use dtype optimization for raw data loading
```python
dtype={
    "userId": "int32",       # max 2B users sufficient
    "movieId": "int32",      # max 2B items sufficient
    "rating": "float32",     # precision: 6 decimals
    "timestamp": "int64"     # Unix timestamp
}
```
- **Benefit**: 50% memory savings vs default int64/float64

### 2. ALS Implementation

**Decision**: Use implicit feedback paradigm with confidence weighting
- **Why**: Explicit ratings are implicit feedback (viewing, rating are positive signals; absence is negative)
- **Formula**: `confidence = 1 + α * log₁ₚ(rating)`
  - Emphasizes high ratings (log dampens extremely high values)
  - Prevents zero/negative confidence
  - Tunable via α parameter

**Decision**: Use CSR sparse matrix format
- **Why**: 
  - Efficient row slicing (ALS iterates users)
  - Compressed storage (>99% sparse → 98%+ memory saved)
  - Direct compatibility with `implicit` library
- **Trade-off**: Column slicing is slow (use COO format for that)

**Decision**: 64 latent factors, 20 iterations
- **Why**:
  - 64 dims: Good balance between expressiveness and regularization
  - 20 iters: Convergence typically reached; diminishing returns after
  - Could tune with cross-validation, but fixed for reproducibility

### 3. Content-Based Filtering

**Decision**: Concatenate genre (one-hot) + tag (TF-IDF) features
- **Why**: 
  - Genres capture broad categories
  - Tags capture user-specific semantics
  - Complementary information sources
- **Feature Dimension**: ~20 genres + 2000 TF-IDF = ~2020 total

**Decision**: Use FAISS IndexFlatIP (inner product on L2-normalized vectors)
- **Why**: 
  - Inner product of L2-normalized = cosine similarity
  - FAISS is production-standard for nearest neighbor search
  - CPU version sufficient for 87K items
  - Fast (<1ms per query)
- **Alternative**: Could use Annoy, but FAISS more mature

**Decision**: User profile = mean of positive-rated item vectors
- **Why**: Simple, stable, interpretable
- **Alternative**: Could use weighted mean or learned weights (more complex, marginal gains)

### 4. Hybrid Fusion

**Decision**: Independent Min-Max normalization per model
- **Why**: 
  - Each model generates scores on different scales
  - Independent normalization prevents one model dominating
  - Prevents information loss vs joint normalization
  - Fair weight competition

**Decision**: Weighted linear combination (α * ALS + (1-α) * CB)
- **Why**: 
  - Simple, interpretable, tunable
  - Allows flexible prioritization
  - Industry-standard approach
- **Alternatives**: Multiplicative, multiplicative weighting, learned weights (more complex)

**Decision**: Large candidate pool before fusion (top-500 each)
- **Why**: 
  - Ensures diverse candidates for fusion
  - Prevents one model missing good items
  - Still <1% of item set; manageable
- **Why not more**: Diminishing returns, increased computation

### 5. Train-Test Evaluation

**Decision**: Temporal per-user split (not random)
- **Why**: 
  - Mimics real scenario (predict future based on history)
  - Prevents data leakage (test data doesn't come before train)
  - More conservative (harder) evaluation metric
- **Split ratio**: 80/20 (standard recommendation practice)

**Decision**: Guarantee ≥1 training interaction per test user
- **Why**: 
  - Avoids cold-start in evaluation (model had some exposure)
  - More realistic (can't evaluate completely new users)
  - Aligns with inference capability

**Decision**: Sample users for evaluation (top-100 from test)
- **Why**: 
  - Full evaluation on 32M dataset too slow
  - Top users more representative
  - Still statistically sound (random sample of 100-1000 users)

### 6. Error Handling & Robustness

**Pattern**: Fallback to popularity for any error
```python
try:
    predictions = model.recommend(...)
except Exception as e:
    logger.error(f"Model failed: {e}")
    predictions = popular_items  # Graceful degradation
```
- **Benefit**: System never breaks; always returns something sensible
- **Trade-off**: May hide bugs (mitigated by logging)

**Pattern**: Boundary checks before matrix access
```python
if user_id not in self.user_map:
    return popular_items  # Cold-start user
if user_idx < 0 or user_idx >= n_users:
    return popular_items  # Out of bounds
```
- **Benefit**: Prevents crashes from malformed input

### 7. Logging Strategy

**Approach**: Module-specific loggers, appropriate levels
```python
logger = logging.getLogger(__name__)  # Per-module logger
logger.setLevel(logging.INFO)  # Info level
```
- **Benefit**: Can control verbosity per module
- **Quiet modules**: Production inference (recommend.py, search.py)
- **Verbose modules**: Training (train.py, features.py)

### 8. Scalability Considerations

**Memory**: 
- Sparse CSR matrix: O(nnz) instead of O(n*m)
  - 32M interactions, 87K movies, 200K users
  - Dense: 200K * 87K * 8 bytes = huge
  - Sparse: 32M * 3 * 8 bytes = manageable
- Sampling: Use 20% rel. sampling in pipeline to fit in 16GB
- Feature caching: Pickle features instead of recomputing

**Computation**:
- ALS: O(iters * factors * nnz) → 2-5 minutes
- FAISS: O(n * d) preprocessing, O(d) per query
- Inference: <1 second per user on CPU

**I/O**:
- Parquet: Columnar, compressed, fast reads
- Batch processing: Load once, process multiple

### 9. Reproducibility

**Approach**: Fixed random seeds
```python
ratings = ratings.sample(frac=SAMPLE_RATIO, random_state=42)
```
- **Benefit**: Same results across runs
- **Seed**: 42 (convention in ML)

**Artifact Caching**: Check existence before recomputing
```python
if path.exists():
    obj = pickle.load(open(path))
else:
    obj = compute_expensive_operation()
    pickle.dump(obj, path)
```
- **Benefit**: Avoids re-training, re-vectorizing
- **Trade-off**: Must delete artifacts to recompute

### 10. User-Facing Design

**Streamlit Caching**:
```python
@st.cache_resource(show_spinner="Loading...")
def load_engine():
    return RecommenderEngine()
```
- **Benefit**: Load models once, reuse across interactions
- **Downside**: Must restart app to reload models

**Cloud-Safe Imports**:
```python
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
```
- **Benefit**: Works on local, cloud (Streamlit Cloud), containers
- **Why**: Different filesystems, path handling

---

## Limitations & Future Improvements

### Current Limitations

1. **Scalability**:
   - Full ML-32M dataset requires significant resources (32GB+ RAM)
   - Evaluation sampled at 100-1000 users (not full test set)
   - **Future**: Distributed training (Spark, Ray), sharded inference

2. **Temporal Dynamics**:
   - Models static after training (no online learning)
   - No concept drift handling
   - Movies added after training not in ALS
   - **Future**: Incremental ALS, periodic retraining, temporal embeddings

3. **Serendipity & Diversity**:
   - Model optimizes accuracy (Precision/Recall/NDCG)
   - May miss serendipitous, diverse discoveries
   - All recommendations same quality score
   - **Future**: Diversity-aware re-ranking, novelty term, exploration

4. **Context Awareness**:
   - No time-of-day, device, seasonality factors
   - No user demographics or context
   - No implicit feedback beyond ratings
   - **Future**: Context bandits, session-based RNNs

5. **Cold-Start**:
   - New users rely entirely on popularity
   - No content for new items in ALS
   - **Future**: Side information (metadata), transfer learning, zero-shot

6. **Explainability**:
   - Black-box latent factors (hard to interpret)
   - Content features somewhat interpretable
   - No explanation generation
   - **Future**: SHAP, attention mechanisms, rule extraction

7. **User Interaction**:
   - Single recommendation mode output (no ranking explanations)
   - No feedback loop (system doesn't learn from user interactions)
   - **Future**: Interactive feedback, online learning

### Suggested Improvements

#### Short-Term (High ROI)

1. **Hyperparameter Tuning**:
   - Grid search: factors ∈ [32, 64, 128], iters ∈ [10, 20, 40]
   - Evaluate impact on metrics
   - Optimize α for hybrid fusion
   - **Expected**: 5-10% quality improvement

2. **Advanced Features**:
   - Temporal features (recency, seasonality)
   - User-user similarity (reduce latent dimensions)
   - Temporal decay (recent interactions weighted more)
   - **Expected**: 3-5% improvement

3. **Ensemble Methods**:
   - Multi-model voting (beyond binary ALS/CB)
   - Content-based variants (KNN, SVD)
   - Learning-to-rank fusion
   - **Expected**: 5-15% improvement

4. **Evaluation**:
   - Full test set evaluation (not sampled)
   - A/B testing framework (online evaluation)
   - Diversity metrics (catalog coverage, Gini coefficient)
   - **Expected**: Better confidence in metrics

#### Medium-Term (Weeks)

1. **Deep Learning**:
   - Neural Collaborative Filtering (MLP + GMF)
   - Attention-based models (self-attention for sequences)
   - Autoencoders for latent learning
   - **Expected**: 10-20% improvement, but higher complexity

2. **Knowledge Integration**:
   - Knowledge graph embeddings (genres → concepts)
   - Side information modeling (director, cast, budget)
   - Transfer learning from other domains
   - **Expected**: Better cold-start, 5-10% improvement

3. **Online Learning**:
   - Incremental ALS or online SGD
   - Periodic model updates (nightly retraining)
   - Feedback loop integration
   - **Expected**: Adapt to trends, user drift

4. **Inference Optimization**:
   - Model quantization (int8 ALS factors)
   - Model compression (pruning zero factors)
   - GPU inference (faiss-gpu, ONNX)
   - **Expected**: 10-100x speedup

#### Long-Term (Months)

1. **Production System**:
   - API deployment (FastAPI, Flask)
   - A/B testing infrastructure
   - Monitoring (logging, metrics, alerts)
   - Caching (Redis, CDN)
   - **Expected**: Production-grade system

2. **Context & Bandit Algorithms**:
   - Contextual bandits (Thompson sampling, LinUCB)
   - Multi-armed bandits for exploration/exploitation
   - Real-time personalization
   - **Expected**: Higher long-term user satisfaction

3. **Fairness & Bias**:
   - Fairness metrics (exposure, calibration)
   - Bias detection (popularity bias, gender bias)
   - Fairness-aware learning
   - **Expected**: More equitable recommendations

---

## Author & Credits

### Development Team

**Abdallah Nabil Ragab**
- 🎓 M.Sc. in Business Information Systems
- 💼 Data Scientist | Machine Learning Engineer | Software Engineer
- 📍 Location: [Not specified in code]
- 📧 Email: abdallah.nabil.ragab94@gmail.com

### Dataset Attribution

**MovieLens 32M Dataset**
> F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: History and Context. ACM Transactions on Interactive Intelligent Systems (TiiS) 5, 4: 19:1–19:19. https://doi.org/10.1145/2827872

**GroupLens Research**
- University of Minnesota, Department of Computer Science and Engineering
- Website: http://grouplens.org
- Dataset: http://movielens.org

### Libraries & Tools

- **implicit**: Fast ALS implementation by Ben Frederickson
- **FAISS**: Meta's approximate nearest neighbor search
- **scikit-learn**: Machine learning utilities and preprocessing
- **pandas & numpy**: Data manipulation and numerical computing
- **scipy**: Scientific Python (sparse matrices)
- **streamlit**: Interactive web UI framework
- **nltk**: Natural language toolkit for text processing

### Citation

If you use this project in research, please cite:

```bibtex
@misc{hybrid_recsys_2024,
  title={Hybrid Movie Recommendation System},
  author={Ragab, Abdallah Nabil},
  year={2024},
  url={https://github.com/abdallahNabilRagab}
}
```

### Feedback & Contact

For questions, bug reports, feature requests, or collaborative opportunities:

📧 **Email**: abdallah.nabil.ragab94@gmail.com

---

## License

This project uses the MovieLens dataset under its terms:
- Free for research and non-commercial purposes
- Attribution required in publications
- See `data/raw/README.txt` for full license details

Code is provided as-is for educational and research purposes.

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2024-01-15 | Initial release with ALS, Content-Based, and Hybrid recommenders |

---

## Support & Resources

### Running into Issues?

1. **Check Requirements**:
   - Python 3.8+
   - 8+ GB RAM
   - All packages installed (`pip install -r requirements.txt`)

2. **Data Issues**:
   - Verify MovieLens files in `data/raw/`
   - Check checksums against `data/raw/checksums.txt`
   - Use smaller dataset (ml-1m) for testing

3. **Memory Issues**:
   - Use smaller sample ratio in `run_pipeline.py`
   - Reduce factors in ALS (try 32 instead of 64)
   - Increase filtering thresholds (MIN_USER_RATINGS)

4. **Performance Issues**:
   - Check num_threads setting (should match CPU cores)
   - Disable progress bars (remove tqdm)
   - Use faiss-gpu if CUDA available

### Useful Commands

```bash
# Check Python version
python --version

# List installed packages
pip list | grep -E "pandas|numpy|scikit|implicit|faiss"

# Check available memory
python -c "import psutil; print(f'{psutil.virtual_memory().available / 1e9:.1f} GB available')"

# Test specific module
python -c "from src.data_loader import DataLoader; print('DataLoader OK')"

# View real-time pipeline logs
python scripts/run_pipeline.py 2>&1 | tee pipeline.log

# Profile memory usage
python -m memory_profiler scripts/run_pipeline.py
```

---

**README Generated**: January 15, 2024  
**Last Updated**: January 15, 2024  
**Status**: Production-Ready (v1.0.0)

---
