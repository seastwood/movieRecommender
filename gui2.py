import pickle
import random

import numpy as np
import pandas as pd
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QVBoxLayout
from matplotlib import pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

print('Loading dataset...')
with open('data4.pkl', 'rb') as f:
    # merged_df, train_df, test_df, movies_df, svd_df, algo = pickle.load(f)
    merged_df, train_df, test_df, movies_df, svd_df, algo = pickle.load(f)
    # merged_df, train_df, test_df, movies_df, algo_sim = pickle.load(f)
    # merged_df, train_df, test_df, movies_df, svd_df = pickle.load(f)

print('Done.')


# Define a function to get the top n recommended movies for a user
def get_top_n_recommendations(user_id, genre, start_year, end_year, seen_rated_checkbox_boolean=False, n=10):
    print('Getting recommendations...')
    print('seen_rated_checkbox_boolean: ', seen_rated_checkbox_boolean)
    if genre == 'All genres':
        # set genre movies to all movies if no genre is selected
        genre_movies = merged_df['movieId'].unique()
    else:
        # Get all movies in the specified genre
        genre_list = genre.split('|')
        genre_movies = movies_df[movies_df['genres'].str.contains(genre_list[0])]
        for g in genre_list[1:]:
            genre_movies = genre_movies[genre_movies['genres'].str.contains(g)]
        genre_movies = genre_movies['movieId'].unique()

    # Get movies that user already rated
    user_movies = merged_df.loc[merged_df['userId'] == user_id, 'movieId']

    # Get movies that user has rated 4 or higher
    # Get the top 100 ratings for the user
    user_top_rated_movies = merged_df.loc[merged_df['userId'] == user_id].nlargest(n, 'rating')

    users_high_ratings_genre_counts = {}
    for movie_genres in movies_df['genres']:
        for genre in movie_genres.split('|'):
            if len(genre.split()) == 1:  # Only count genres with a single word
                users_high_ratings_genre_counts[genre] = 0
    for movie_genres in user_top_rated_movies['genres']:
        for genre in movie_genres.split('|'):
            if len(genre.split()) == 1:  # Only count genres with a single word
                if genre in users_high_ratings_genre_counts:
                    users_high_ratings_genre_counts[genre] += 1
                else:
                    users_high_ratings_genre_counts[genre] = 1

    print(users_high_ratings_genre_counts)

    # Get movies in the specified genre that the user hasn't rated
    if seen_rated_checkbox_boolean:
        unseen_genre_movies = set(genre_movies)
    else:
        unseen_genre_movies = set(genre_movies) - set(user_movies)

    # Prepare the testset
    testset = [[user_id, movie_id, 0] for movie_id in unseen_genre_movies]

    # Predict the ratings for the unseen movies
    predictions = svd_df.test(testset)

    # Extract the predicted ratings for each movie
    recommendations = [(pred.iid, pred.est) for pred in predictions]

    # Convert recommendations to a pandas dataframe
    recommendations_df = pd.DataFrame(recommendations, columns=['movieId', 'predicted_rating'])
    # Merge the movies dataframe with the recommendations dataframe
    recommended_movies = pd.merge(recommendations_df, movies_df, on='movieId')

    print('Extracting Year info...')
    # Extract the year information from the movie title
    year_regex = r"\((\d{4})\)$"
    recommended_movies["year"] = recommended_movies["title"].str.extract(year_regex, expand=True)
    recommended_movies["year"] = pd.to_numeric(recommended_movies["year"], errors="coerce")

    print('Grabbing movies within year range...')
    # print("FINDING MOVIES IN YEAR: ", int(year_range))
    # Filter movies based on the year range
    if start_year - end_year == 0:
        recommended_movies = recommended_movies[recommended_movies["year"] == start_year]
        print("FINDING MOVIES IN YEAR: ", start_year)
    else:
        recommended_movies = recommended_movies[(recommended_movies["year"] >= start_year) &
                                                (recommended_movies["year"] <= end_year)]
        print("FINDING MOVIES IN YEAR RANGE: ", start_year, ' - ', end_year)

    # Count the number of movies in each genre that the user has watched
    genre_counts = recommended_movies.explode('genres').groupby('genres').size()

    # Check if the user has rated each recommended movie
    rated_movies = set(merged_df[merged_df['userId'] == user_id]['movieId'])
    recommended_movies['rated'] = recommended_movies['movieId'].isin(rated_movies)

    return recommended_movies.sort_values('predicted_rating', ascending=False).head(
        n), genre_counts, users_high_ratings_genre_counts


def count_user_genre_views(user_id):
    # Get all movies that the user has watched
    user_movies = merged_df.loc[merged_df['userId'] == user_id, 'movieId']
    # Get the genres for each of the user's watched movies
    user_genres = movies_df[movies_df['movieId'].isin(user_movies)]['genres'].str.split('|').explode()
    # Count the number of movies in each genre that the user has watched
    genre_counts = user_genres.value_counts().to_dict()
    return genre_counts


class MainWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.seen_rated_checkbox = None
        self.end_year_input = None
        self.start_year_input = None
        self.genre_counts_label = None
        self.recommendations_label = None
        self.bar_graph_checkbox = None
        self.genre_counts_list = None
        self.recommendations_list = None
        self.n_input = None
        self.user_id_input = None
        self.genre_input = None  # Add a new input field for the genre
        self.bar_graph_figure = None  # Add a variable to hold the figure for the bar graph
        self.init_ui()

        # Set the window title
        self.setWindowTitle("Movie Recommender")

    def init_ui(self):

        # Get the max and min user ID
        max_user_id = merged_df['userId'].max() + 1
        min_user_id = merged_df['userId'].min()

        # Set up the user ID input box
        user_id_label = QtWidgets.QLabel("User ID: max user id = {}".format(max_user_id))
        self.user_id_input = QtWidgets.QSpinBox()
        self.user_id_input.setMinimum(int(min_user_id))
        self.user_id_input.setMaximum(int(max_user_id))
        self.user_id_input.setFixedWidth(450)

        # Set up the randomizer button
        random_button = QtWidgets.QPushButton("Randomize User ID")
        random_button.clicked.connect(self.set_random_user_id)

        # Create the layout
        user_id_layout = QtWidgets.QHBoxLayout()
        user_id_layout.addWidget(user_id_label)
        user_id_layout.addWidget(self.user_id_input)
        user_id_layout.addWidget(random_button)
        user_id_layout.addSpacing(157)

        # Set up the number of recommendations input box
        n_label = QtWidgets.QLabel("Number of recommendations:")
        self.n_input = QtWidgets.QSpinBox()
        self.n_input.setMinimum(1)
        self.n_input.setMaximum(100)
        n_layout = QtWidgets.QHBoxLayout()
        n_layout.addWidget(n_label)
        n_layout.addWidget(self.n_input)
        self.n_input.setFixedWidth(300)

        # Extract the year information from the movie titles using regular expressions
        years = movies_df['title'].str.extract(r"\((\d{4})\)$", expand=True)

        # Convert the extracted year values to integers
        years = pd.to_numeric(years[0], errors='coerce')

        # Set up the year range inputs
        year_range_layout = QtWidgets.QVBoxLayout()
        year_range_label = QtWidgets.QLabel("Year Range:")
        year_range_layout.addWidget(year_range_label)

        start_year_label = QtWidgets.QLabel("Start Year: minimum = {}".format(int(years.min())))
        self.start_year_input = QtWidgets.QSpinBox()
        self.start_year_input.setMinimum(int(years.min()))
        self.start_year_input.setMaximum(int(years.max()))
        self.start_year_input.setValue(int(years.min()))
        start_year_layout = QtWidgets.QHBoxLayout()
        start_year_layout.addWidget(start_year_label)
        start_year_layout.addWidget(self.start_year_input)
        year_range_layout.addLayout(start_year_layout)
        self.start_year_input.setFixedWidth(100)

        # Set up the end year input
        end_year_label = QtWidgets.QLabel("End Year: maximum = {}".format(int(years.max())))
        self.end_year_input = QtWidgets.QSpinBox()
        self.end_year_input.setMinimum(int(years.min()))
        self.end_year_input.setMaximum(int(years.max()))
        self.end_year_input.setValue(int(years.max()))
        end_year_layout = QtWidgets.QHBoxLayout()
        end_year_layout.addWidget(end_year_label)
        end_year_layout.addWidget(self.end_year_input)
        year_range_layout.addLayout(end_year_layout)
        self.end_year_input.setFixedWidth(100)

        # Connect the on_start_year_changed slot to the valueChanged signal of self.start_year_input
        self.start_year_input.valueChanged.connect(self.on_start_year_changed)

        # Set up the genre input dropdown
        genre_label = QtWidgets.QLabel("Genre:")
        self.genre_input = QtWidgets.QComboBox()
        self.genre_input.addItem("All genres")
        genres = movies_df["genres"].unique()

        # Define sorting function
        def genre_sort_key(genre):
            return genre.count('|'), genre

        # Sort genres by number of '|' characters and then alphabetically
        genres = sorted(genres, key=genre_sort_key)

        self.genre_input.addItems(genres)

        genre_layout = QtWidgets.QHBoxLayout()
        genre_layout.addWidget(genre_label)
        genre_layout.addWidget(self.genre_input)

        # Set up the submit button
        submit_button = QtWidgets.QPushButton("Get recommendations")
        submit_button.clicked.connect(self.get_recommendations)

        self.recommendations_label = QtWidgets.QLabel('Top Recommendations')
        self.recommendations_list = QtWidgets.QTableWidget()
        self.recommendations_list.setColumnCount(4)
        self.recommendations_list.setHorizontalHeaderLabels(['Movie', 'Genres', 'Predicted Rating', 'Seen/Rated'])
        self.recommendations_list.horizontalHeader().setStretchLastSection(True)
        self.recommendations_list.verticalHeader().setVisible(False)
        self.recommendations_list.setShowGrid(False)

        self.recommendations_list.setColumnWidth(0, 300)  # title
        self.recommendations_list.setColumnWidth(1, 150)  # genres
        self.recommendations_list.setColumnWidth(2, 100)  # rating
        self.recommendations_list.setColumnWidth(3, 100)  # rated

        recommendations_layout = QtWidgets.QVBoxLayout()
        recommendations_layout.addWidget(self.recommendations_label)
        recommendations_layout.addWidget(self.recommendations_list)

        checkboxes_layout = QtWidgets.QHBoxLayout()
        self.bar_graph_checkbox = QtWidgets.QCheckBox("Generate bar graph")
        self.seen_rated_checkbox = QtWidgets.QCheckBox("Include seen/rated movies")
        # self.content_filter_checkbox = QtWidgets.QCheckBox("User Preference Content Filtering")
        checkboxes_layout.addWidget(self.bar_graph_checkbox)
        checkboxes_layout.addWidget(self.seen_rated_checkbox)
        # checkboxes_layout.addWidget(self.content_filter_checkbox)
        checkboxes_layout.addSpacing(600)

        self.genre_counts_label = QtWidgets.QLabel('Genre Counts')
        self.genre_counts_list = QtWidgets.QTableWidget()
        self.genre_counts_list.setColumnCount(2)
        self.genre_counts_list.setHorizontalHeaderLabels(['Genre', 'Count'])
        self.genre_counts_list.horizontalHeader().setStretchLastSection(True)
        self.genre_counts_list.verticalHeader().setVisible(False)
        self.genre_counts_list.setShowGrid(False)

        self.genre_counts_list.setColumnWidth(0, 100)
        self.genre_counts_list.setColumnWidth(1, 100)
        self.genre_counts_list.setMinimumWidth(205)
        self.genre_counts_list.setMaximumWidth(205)

        genre_counts_layout = QtWidgets.QVBoxLayout()
        genre_counts_layout.addWidget(self.genre_counts_label)
        genre_counts_layout.addWidget(self.genre_counts_list)

        # Create a QHBoxLayout and add both QTableWidgets to it
        table_layout = QtWidgets.QHBoxLayout()
        table_layout.addLayout(recommendations_layout)
        table_layout.addLayout(genre_counts_layout)

        self.bar_graph_figure = plt.figure()  # Create a new figure for the bar graph
        # Create a QHBoxLayout and add both QTableWidgets and the figure canvas to it
        graph_layout = QtWidgets.QHBoxLayout()
        graph_layout.addWidget(self.recommendations_list)
        graph_layout.addWidget(self.genre_counts_list)
        graph_layout.addWidget(FigureCanvas(self.bar_graph_figure))  # Add the figure canvas to the layout

        # Set fixed width of graph layout
        graph_layout_wrapper = QtWidgets.QWidget()
        graph_layout_wrapper.setLayout(graph_layout)
        graph_layout_wrapper.setFixedWidth(930)

        options_layout1 = QtWidgets.QVBoxLayout()
        options_layout1.addLayout(n_layout)
        options_layout1.addLayout(genre_layout)

        options_layout2 = QtWidgets.QVBoxLayout()
        options_layout2.addLayout(year_range_layout)

        # Create a new QHBoxLayout and add the options layouts to it
        options_h_layout = QtWidgets.QHBoxLayout()
        options_h_layout.addLayout(options_layout1)
        options_h_layout.addSpacing(50)  # add a 20 pixel spacer
        options_h_layout.addLayout(options_layout2)
        options_h_layout.addSpacing(140)

        # Add the QHBoxLayout and year_range_layout to all_options_layout
        all_options_layout = QtWidgets.QVBoxLayout()
        all_options_layout.addLayout(options_h_layout)
        all_options_layout.addLayout(year_range_layout)

        # Set up the main layout and add the input boxes, submit button, and the QHBoxLayout to it
        main_layout = QVBoxLayout()
        main_layout.addLayout(user_id_layout)
        main_layout.addLayout(all_options_layout)
        main_layout.addLayout(checkboxes_layout)
        main_layout.addWidget(submit_button)
        main_layout.addLayout(table_layout)
        main_layout.addWidget(graph_layout_wrapper)

        self.setLayout(main_layout)

        # Set the default size of the main window
        self.setGeometry(100, 100, 930, 1200)

        self.show()

    def set_random_user_id(self):
        max_user_id = self.user_id_input.maximum()
        random_user_id = random.randint(1, max_user_id)
        self.user_id_input.setValue(random_user_id)

    def on_start_year_changed(self):
        self.end_year_input.setMinimum(self.start_year_input.value())

    def generate_bar_graph(self, genre_counts, users_high_ratings_genre_counts, n):
        print('GENERATING BAR GRAPH')
        if self.bar_graph_checkbox.isChecked():
            self.bar_graph_figure.clf()  # Clear the figure before creating the new bar graph
            ax = self.bar_graph_figure.add_subplot(111)  # Create a new subplot on the figure
            xticks = range(len(genre_counts))
            ax.set_xticks(xticks)
            ax.set_xticklabels(genre_counts.keys(), rotation=45, ha='right')  # Set the x-axis labels
            # Create the bar graph
            bar_width = 0.4  # Set the width of each bar
            ax.bar(xticks, genre_counts.values(), width=bar_width, label='Recommended Genre Count')
            ax.bar([x + bar_width for x in xticks], users_high_ratings_genre_counts.values(),
                   width=bar_width, label=('Users Top {} Seen/Rated'.format(n)))
            ax.set_title('Top {} Recommended Movie Genre Counts Compared with Users Top {} Seen/Rated Movies'
                         .format(n, n))
            ax.set_xlabel('Genre')
            ax.set_ylabel('Count / Other Data')
            ax.legend()  # Show a legend for the two sets of data
            self.bar_graph_figure.tight_layout()  # Adjust the layout to avoid overlapping labels
            self.bar_graph_figure.canvas.draw()  # Redraw the canvas to update the figure

    # Generate the bar graph code here

    def get_recommendations(self):
        # Get the user ID and number of recommendations from the input boxes
        user_id = int(self.user_id_input.text())
        n = self.n_input.value()
        genre = self.genre_input.currentText()  # Get the selected genre from the dropdown menu

        print('Setting start and end years...')
        start_year = int(self.start_year_input.text())
        end_year = int(self.end_year_input.text())
        print('year range: ', start_year, ' - ', end_year)

        print('Setting year range...')
        # year_range = range(start_year, end_year)

        # Get the top n recommended movies for the user
        print('Calling method for recommendations...')
        recommended_movies, genre_counts, users_high_ratings_genre_counts = \
            get_top_n_recommendations(user_id, genre, start_year, end_year,
                                      self.seen_rated_checkbox.isChecked(), n=n)
        print('Recommendations retrieved...')

        # Update the recommendations list widget with the max user ID and recommended movies
        self.recommendations_list.setRowCount(len(recommended_movies))

        # Count the number of movies in each genre
        genre_counts = {}
        for movie_genres in movies_df['genres']:
            for genre in movie_genres.split('|'):
                if len(genre.split()) == 1:  # Only count genres with a single word
                    genre_counts[genre] = 0
        for movie_genres in recommended_movies['genres']:
            for genre in movie_genres.split('|'):
                if len(genre.split()) == 1:  # Only count genres with a single word
                    if genre in genre_counts:
                        genre_counts[genre] += 1
                    else:
                        genre_counts[genre] = 1

        print('Completed Loop of genre counts')
        print('GENRE COUNTS')
        print(genre_counts)
        print('users_high_ratings_genre_counts')
        print(users_high_ratings_genre_counts)
        if self.bar_graph_checkbox.isChecked():
            sorted_genre_counts = dict(sorted(genre_counts.items()))
            self.generate_bar_graph(sorted_genre_counts, users_high_ratings_genre_counts, n)

        # Sort the genre_counts dictionary by count in descending order
        genre_counts = dict(sorted(genre_counts.items(), key=lambda x: x[1], reverse=True))

        # Update the genre counts table
        self.genre_counts_list.setRowCount(len(genre_counts))
        row = 0
        for genre, count in genre_counts.items():
            self.genre_counts_list.setItem(row, 0, QtWidgets.QTableWidgetItem(genre))
            self.genre_counts_list.setItem(row, 1, QtWidgets.QTableWidgetItem(str(count)))
            row += 1

        for row, (_, movie) in enumerate(recommended_movies.iterrows()):
            # print('Setting title...')
            title = movie['title']
            # print('Setting rating...')
            rating = movie['predicted_rating']
            # print('Setting genres...')
            genres = movie['genres']
            # print('Setting rated/seen...')
            rated = 'yes' if movie['rated'] else 'no'

            title_item = QtWidgets.QTableWidgetItem(title)
            genres_item = QtWidgets.QTableWidgetItem(genres)
            rating_item = QtWidgets.QTableWidgetItem(f"{rating:.2f}")
            rated_item = QtWidgets.QTableWidgetItem(rated)

            self.recommendations_list.setItem(row, 0, title_item)
            self.recommendations_list.setItem(row, 1, genres_item)
            self.recommendations_list.setItem(row, 2, rating_item)
            self.recommendations_list.setItem(row, 3, rated_item)


if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    window = MainWindow()
    app.exec_()
