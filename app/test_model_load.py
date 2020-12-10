from fastai.vision.all import *
load_path = Path(__file__).parent
classifier_save_filename = 'subculture_classifier.pkl'
loaded_subculture_learn = load_learner(load_path/classifier_save_filename)

def get_formatted_prediction(img_to_predict):
  prediction,p_index,probabilities = loaded_subculture_learn.predict(img_to_predict)
  return f"prediction: {prediction} probability: {probabilities[p_index]}"

dj_prediction = get_formatted_prediction(load_path/'david_and_julie.jpg')
jh_prediction = get_formatted_prediction(load_path/'james_hoffmann.jpg')

print(f"\nloading categories...\n{loaded_subculture_learn.dls.vocab}")
print(f"\npredicting james hoffmann...\n{jh_prediction}")
print(f"\npredicting david and Julie...\n{dj_prediction}\n")
