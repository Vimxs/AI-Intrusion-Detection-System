def predict_sample(model, sample):
    prediction = model.predict([sample])
    return prediction