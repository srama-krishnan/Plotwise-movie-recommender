# Tamil Movie Recommender 🎬 powered by OpenAI Embeddings & Atlas

This smart recommender suggests similar Tamil movies based on plot similarity using OpenAI’s embedding API. It also maps the entire movie embedding space with Nomic Atlas for interactive exploration.

---

## 🔍 Features

- Embeds 1500+ recent Tamil movie plots using `text-embedding-3-small`
- Finds nearest neighbors via cosine distance
- Visualizes all embeddings on Nomic Atlas
- Caches all embeddings to avoid recomputing
- Outputs clear movie recommendations (Title + Genre)

---

## 📂 Project Structure

movie-recommender/  
├── app.py                   # Main logic  
├── movie_plots.csv          # Dataset  
├── movie_embeddings.pkl     # Cached embeddings  
├── .env                     # Stores OpenAI API Key  
├── requirements.txt  

---

## 🧪 Sample Outputs

## **Atlas Visualization**  

![image](https://github.com/user-attachments/assets/f69dd976-6f69-4447-9dad-f8199369e97f)


## **Recommendation for: Velaikkaran (thriller)**  

![image](https://github.com/user-attachments/assets/4c9a8a53-ca29-45f9-bdb2-ff7383c0bf60)


## **Recommendation for: Baahubali: The Beginning (epic / history)**  

![image](https://github.com/user-attachments/assets/e2d1e863-11c7-436a-8b91-09b0957cf472)

---

## 💻 How to Run

1. **Install dependencies**
   ```
   pip install -r requirements.txt
   ```

2. **Set your OpenAI key**  
   In `.env` file:
   ```
   APIKEY = sk-xxxx...
   ```

3. **Run the script**
  ```
   python app.py
```

   It will:
   - Embed the plots using OpenAI API
   - Cache and reuse embeddings
   - Visualize using Atlas
   - Print top 5 movie suggestions for the given one

---

## 📌 Notes

- You need an OpenAI API key to generate embeddings
- Embeddings are saved in a pickle file for re-use
- Login to Nomic once via CLI using:

  nomic login

- Cost of embedding is approximately `$0.02 per 1000 plots (1k tokens each)`

---

## 📜 License

MIT License © 2025  
Author: Ramakrishnan S

---

