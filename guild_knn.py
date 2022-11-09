from surprise import Dataset, KNNBasic

def train():

    # Load the movielens-100k dataset
    data = Dataset.load_builtin("ml-100k")

    # Retrieve the trainset.
    trainset = data.build_full_trainset()

    # Build an algorithm, and train it.
    algo = KNNBasic()
    algo.fit(trainset)

    return algo

def predict(uid, iid, r_ui, algo):

    # get a prediction for specific users and items.
    pred = algo.predict(uid, iid, r_ui=4, verbose=True)

uid = 196
iid = 302
r_ui = 4

algo = train()
predict(uid, iid, r_ui, algo)

