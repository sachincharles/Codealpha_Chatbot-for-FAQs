import string
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nlp = spacy.load("en_core_web_sm")

faqs = {
    "What is AI?": " AI is the simulation of human intelligence in machines",
    "What are the main types of AI?": " There are Narrow AI (weak), General AI (strong), and Superintelligent AI",
    "How does AI learn?": "AI learns through machine learning, deep learning, and other algorithms.",
    "What can AI do?": "AI can perform tasks like image recognition, natural language processing, and data analysis",
    "What are the ethical concerns of AI? ": "Ethical concerns include bias, privacy, and job displacement. ",
   
}

questions = list(faqs.keys())
answers = list(faqs.values())

def preprocess_with_spacy(text):
    doc = nlp(text.lower().translate(str.maketrans("", "", string.punctuation)))
    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    return " ".join(tokens)


processed_questions = [preprocess_with_spacy(q) for q in questions]

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(processed_questions)

def get_response(user_input):
    user_cleaned = preprocess_with_spacy(user_input)
    user_vector = vectorizer.transform([user_cleaned])
    
    similarity = cosine_similarity(user_vector, tfidf_matrix)
    best_index = similarity.argmax()
    best_score = similarity[0][best_index]
    
    if best_score > 0.3:
        return answers[best_index]
    else:
        return "Sorry, I couldn't find a relevant answer."


import tkinter as tk
from tkinter import scrolledtext
from PIL import Image, ImageTk

def send_message():
    user_input = user_entry.get()
    if user_input.strip() == "":
        return
    chat_area.config(state='normal')
    chat_area.insert(tk.END, "You: " + user_input + "\n")
    response = get_response(user_input)
    chat_area.insert(tk.END, "Bot: " + response + "\n\n")
    chat_area.config(state='disabled')
    chat_area.yview(tk.END)
    user_entry.delete(0, tk.END)

if __name__ == "__main__":
    root = tk.Tk()
    root.title("FAQ Chatbot using SpaCy")
    root.geometry("500x500")

    # Load and set background image
    try:
        bg_image = Image.open("your_image.png")  # <-- Replace with your image filename
        try:
            bg_image = bg_image.resize((500, 500), Image.Resampling.LANCZOS)
        except AttributeError:
            bg_image = bg_image.resize((500, 500), Image.ANTIALIAS)
        bg_photo = ImageTk.PhotoImage(bg_image)
        bg_label = tk.Label(root, image=bg_photo)
        bg_label.image = bg_photo  # Keep a reference
        bg_label.place(x=0, y=0, relwidth=1, relheight=1)
    except Exception as e:
        print("Background image could not be loaded:", e)

    # Place chat area on top of background
    chat_area = scrolledtext.ScrolledText(
        root,
        wrap=tk.WORD,
        state='disabled',
        font=("Arial", 12),
        bg="#1a1a1a",  # Use a color from your image
        fg="white",
        insertbackground="white"
    )
    chat_area.place(relx=0.02, rely=0.18, relwidth=0.96, relheight=0.65)

    # Place entry and button on top of background
    user_entry = tk.Entry(root, font=("Arial", 12), bg="black", fg="white", insertbackground="white")
    user_entry.place(relx=0.02, rely=0.86, relwidth=0.75, relheight=0.08)
    user_entry.focus()

    send_button = tk.Button(root, text="Send", command=send_message, font=("Arial", 12), bg="gray20", fg="white", activebackground="gray30", activeforeground="white")
    send_button.place(relx=0.78, rely=0.86, relwidth=0.20, relheight=0.08)

    def on_enter(event):
        send_message()

    user_entry.bind('<Return>', on_enter)

    chat_area.config(state='normal')
    chat_area.tag_configure('center', justify='center')
    chat_area.tag_configure('welcome', justify='center', font=("Arial", 14, "bold"), foreground="white")
    chat_area.insert(tk.END, "\n\n\n\n\n\n\n\n\n\n")  # Add vertical space to push to middle
    chat_area.insert(tk.END, "FAQ Chatbot using SpaCy To learn AI\n(Enter the prompt below)\n\n", ('center', 'welcome'))
    chat_area.config(state='disabled')

    root.mainloop()