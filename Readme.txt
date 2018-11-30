Fine grained Classification

1.Models
    model_OSME_miru.py -> Attention branch + OSME
    model_OSME_alpha.py -> OSME alpha
    model_with_OSME_SE.py -> OSME
    model_OSME_miru_dog.py -> Attention branch + OSME with StandfordDogs dataset
    
2. How to use
    Just run the python scripts.
    To change models or other things rewrite the code.
    
3.Using ImageDataGenerator.flow_from_directory:
        example:
        
train_datagen = ImageDataGenerator(rescale = 1./255, 
                                   zoom_range=[0.6,1],
                                   rotation_range=30,
                                   horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale = 1./255)

train_generator = train_datagen.flow_from_directory(
        train_path,
        target_size=(img_size, img_size),
        batch_size=BATCH_SIZE,
        seed = 13,
        multi_outputs=True,  # If the model has multi outputs: True, else None.(default is None)
        out_n = 2 # write the number of outputs that the model has here.
        )

validation_generator = test_datagen.flow_from_directory(
        test_path,
        target_size=(img_size, img_size),
        batch_size=BATCH_SIZE,
        seed = 13,
        multi_outputs=True,  # If the model has multi outputs: True, else None.(default is None)
        out_n = 2 # write the number of outputs that the model has here.
        )
    
    