from flask import Flask, render_template, request
import comment_loader
from urllib.parse import urlparse, parse_qs
import pandas as pd
import process
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import pickle
from wordcloud import WordCloud

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('home.html', result=None)

@app.route('/process_form', methods=['POST'])
def process_form():
    if request.method == 'POST':
        youtube_url = request.form.get('input_text')
        # Perform sentiment analysis or any other processing here
        # For now, let's just return the URL as the result
        video_id = extract_video_id(youtube_url)
        df=comment_loader.get_comments(video_id)
        # result = f" {video_id}"
        df.to_csv('comments.csv')
        # **********************************************
        # process form and give for further analusis
        df=pd.read_csv('comments.csv')
        newdf= process.preprocess(df)
        with open('tfidf_vectorizer (1).pkl', 'rb') as file:
            tfidf = pickle.load(file)
        X_tfidf = tfidf.transform(newdf).toarray()
        with open('logistic_model (1).pkl', 'rb') as file:
            model = pickle.load(file)
        df['sentiment'] = model.predict(X_tfidf)
        df.to_csv('commentsentiment.csv')
        # ************************************************
        return render_template('home.html',result=True)
    
@app.route('/get_analysis', methods=['POST'])
def get_analysis():
    if request.method == 'POST':
        # youtube_url = request.form.get('input_text')
        # video_id = extract_video_id(youtube_url)
        # df=comment_loader.get_comments(video_id)
        # result = f" {video_id}"
        # /////////////////////
        # df=pd.read_csv('comments.csv')
        # newdf= process.preprocess(df)
        # with open('tfidf_vectorizer (1).pkl', 'rb') as file:
        #     tfidf = pickle.load(file)
        # X_tfidf = tfidf.transform(newdf).toarray()
        # with open('logistic_model (1).pkl', 'rb') as file:
        #     model = pickle.load(file)
        # df['sentiment'] = model.predict(X_tfidf)
        # df.to_csv('commentsentiment.csv')
        # /////////////////////////////
        # Pass DataFrame directly to the template
        dfd=pd.read_csv('commentsentiment.csv')
        return render_template('analysis.html', result=dfd)


def extract_video_id(url):
    parsed_url = urlparse(url)

    if parsed_url.netloc == 'youtu.be':
        # If the URL is of the shortened format youtu.be/{video_id}
        return parsed_url.path[1:]
    elif parsed_url.netloc == 'www.youtube.com' and parsed_url.path == '/watch':
        # If the URL is of the regular format www.youtube.com/watch?v={video_id}
        query_params = parse_qs(parsed_url.query)
        return query_params['v'][0] if 'v' in query_params else None
    else:
        # URL format not supported
        return None




def generate_pie_chart(df):
    # Count the occurrences of each sentiment value
    sentiment_counts = df['sentiment'].value_counts()
    
    # Calculate percentages
    total = sentiment_counts.sum()
    positive_percentage = sentiment_counts.get(1, 0) / total * 100
    neutral_percentage = sentiment_counts.get(0, 0) / total * 100
    negative_percentage = sentiment_counts.get(-1, 0) / total * 100

    # Create pie chart
    labels = ['Positive', 'Neutral', 'Negative']
    sizes = [positive_percentage, neutral_percentage, negative_percentage]
    colors = ['#87CEFA', '#FFA07A', '#D3D3D3']
    
    plt.figure(figsize=(10,8), dpi=80)
    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.legend(
          title="Sentiments",
          loc="best",
          )
    # Convert plot to bytes buffer
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    
    # Encode the image to base64
    image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    plt.close()  # Close the plot to release memory
    
    return image_base64
def generate_bar_plot(df):
    # Count the occurrences of each sentiment value
    sentiment_counts = df['sentiment'].value_counts()

    # Get the counts of positive, negative, and neutral sentiments
    positive_count = sentiment_counts.get(1, 0)
    neutral_count = sentiment_counts.get(0, 0)
    negative_count = sentiment_counts.get(-1, 0)

    # Create bar plot
    labels = ['Positive', 'Neutral', 'Negative']
    counts = [positive_count, neutral_count, negative_count]

    plt.figure(figsize=(10, 6))
    plt.bar(labels, counts, color=['#87CEFA', '#D3D3D3', '#FFA07A'])

    # Add labels and title
    plt.xlabel('Sentiment')
    plt.ylabel('Count')
    plt.title('Sentiment Distribution')

    # Convert plot to bytes buffer
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)

    # Encode the image to base64
    image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    plt.close()  # Close the plot to release memory

    return positive_count,neutral_count,negative_count,image_base64

@app.route('/visualize_chart', methods=['POST'])
def visualize_chart():
    if request.method == 'POST':
        dff =pd.read_csv('commentsentiment.csv')
        pie_image_base64 = generate_pie_chart(dff)
        
        # Generate bar plot
        positive_count,neutral_count,negative_count,bar_image_base64 = generate_bar_plot(dff) 
        
        # Pass the base64 encoded image to the template
        return render_template('combined_charts.html',positive_count=positive_count,neutral_count=neutral_count,negative_count=negative_count, pie_image=pie_image_base64, bar_image=bar_image_base64)
    

def generate_word_cloud(text):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')

    # Convert plot to bytes buffer
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)

    # Encode the image to base64
    image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

    plt.close()
    return image_base64

@app.route('/generate_wordcloud/<sentiment>')
def generate_wordcloud(sentiment):
    # Load the dataset
    df = pd.read_csv('commentsentiment.csv')

    # Filter the dataset based on sentiment
    if sentiment == 'positive':
        text = ' '.join(df[df['sentiment'] == 1]['text'].values)
    elif sentiment == 'neutral':
        text = ' '.join(df[df['sentiment'] == 0]['text'].values)
    elif sentiment == 'negative':
        text = ' '.join(df[df['sentiment'] == -1]['text'].values)
    else:
        return 'Invalid sentiment'

    image_base64 = generate_word_cloud(text)
    return render_template('wordcloud.html', image=image_base64)

@app.route('/wordcloud', methods=['POST'])
def wordcloud():
    if request.method == 'POST':
        # Logic to generate and display word cloud
        return render_template('wordcloud.html')
    

if __name__ == '__main__':
    app.run(debug=True)
