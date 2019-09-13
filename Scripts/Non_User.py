import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ratings = pd.read_csv('../data/ratings.csv')
books_data = pd.read_csv('../data/books.csv')

books = pd.DataFrame(books_data, columns=['book_id', 'authors', 'title', 'average_rating','ratings_count', 'image_url'])
books_info = pd.merge(books, ratings, on='book_id')

book_rating_matrix = pd.pivot_table(books_info, index='user_id', values='rating', columns='book_id',fill_value=0)
book_corr = np.corrcoef(book_rating_matrix.T)

book_list =  list(book_rating_matrix)
book_titles =[] 
for i in range(len(book_list)):
    book_titles.append(book_list[i])

input_book = input("Enter a book name:\n")
book_entered = books_data.loc[books_data['original_title'] == input_book]
book_entered_id = book_entered['id']
book = book_entered_id.values
book_index = book_titles.index(book[0])
corr_score = book_corr[book_index]
condition = (corr_score >= 0.5)
correlated_books = np.extract(condition, book_titles)

similar_book_titles = []
for i in correlated_books:
    title = books_data.loc[books_data['id'] == i]
    similar_book_titles.append(title['title'])

title = book_entered['title']
print("The Book you entered is\n", title) 
print("Books you may like are:")

for i in range(len(similar_book_titles)):
    #if
    print(similar_book_titles[i].to_string(), sep = "\n") 
