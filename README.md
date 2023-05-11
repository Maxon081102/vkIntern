# vkIntern

**Веса для glove https://huggingface.co/stanfordnlp/glove/resolve/main/glove.840B.300d.zip**

Моя модель работает так: берет пользователя, выбирает два фильма для подсчета, потом строит эмбеддинг пользователя без учета этих фильмов

Далее считает для двух фильмов эмбеддинги(выбирает поровну пользователей с рейтингом для фильма больше 3 и меньше 3, потом для них считает эмбеддинги, как в самом начале, а потом, на основе эмбеддингов пользователей считает эмбеддинг фильма)

Считаю лосс так $$Loss = (d(userEmb, firstMovieEmb) - d(userEmb, secondMovieEmb) + firstMovieRating - secondMovieRating)^2$$

d - евклидово расстояние

Интуиция такая: сделать так, чтобы $$d(userEmb, movieEmb) = const + 5 - movieRating$$

Метрика у меня такая $$5 - d(userEmb, movieEmb) - movieRating$$


