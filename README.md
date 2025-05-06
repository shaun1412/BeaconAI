# BeaconAI: Hybrid Event Recommendation System

**BeaconAI** is a hybrid recommendation engine that leverages both collaborative and content-based filtering to suggest personalized events to users. It is built using the [LightFM](https://making.lyst.com/lightfm/docs/home.html) library and simulates user-event interactions with rich feature metadata.

## 🚀 Features

- Hybrid recommendation using **LightFM** with **WARP loss**
- Integrates **user and item metadata** (age groups, event types, time preferences, locations)
- Generates synthetic interaction data for testing
- Personalized **top-N event recommendations** with interpretable outputs
- End-to-end pipeline: dataset preparation → model training → evaluation

## 🛠️ Technologies Used

- Python
- NumPy
- LightFM
- Random sampling (for data simulation)

## 📦 Project Structure

- `BeaconAI` class: handles dataset construction, model training, and inference
- Synthetic dataset generation for users, events, and interactions
- Evaluation pipeline for inspecting recommended vs. previously liked events

## 🧪 Example Output

For a sample user, the system displays:

- The features of the user
- Events the user has liked
- Features of those liked events
- Top 5 recommended events not previously liked
- Feature breakdown of recommended events

## 🧠 How It Works

1. Users and events are assigned random feature tags from predefined pools.
2. Interactions are simulated by randomly assigning event likes to users.
3. LightFM learns latent embeddings from the interaction matrix and feature metadata.
4. The trained model predicts the most relevant events for each user.

## 📌 Getting Started

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install lightfm numpy

## Running the Script 
python beacon_ai.py

