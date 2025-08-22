Proiect CNN pentru MNIST

----Un proiect simplu de clasificare a cifrelor MNIST folosind rețele neuronale convoluționale (CNN).

Structura:
    data_loader.py - Încarcă și preprocesează datele
    model.py - Construiește arhitectura CNN
    train.py - Antrenează modelul
    evaluate.py - Evaluează performanța

Cum se folosește:
    Instalează cerințele: pip install tensorflow scikit-learn matplotlib numpy
    Rulează: python train.py
    Evaluează: python evaluate.py

Optiuni avansate:
    hyperparam_tuning.py - Găsește parametrii optimi (rulează separat)

Ieșire:
    Modelul salvat în best_model.h5
    Metricile de performanță afișate în consolă

