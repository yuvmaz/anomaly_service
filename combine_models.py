import keras
import os 
import glob
import pickle

data = {}

for hdf_file in glob.glob("*.h5"):
    set_name, _ = hdf_file.split('.', 1)
    params_file = set_name + '.pkl'
    for this_file in [params_file]:
        if not os.path.exists(params_file):
            print("File {} exists but could not find file {}".format(hdf_file, this_file))
            continue
    else:
        print("Located {} and {} ...".format(hdf_file, params_file))

    print("Processing {}...", hdf_file)
    model = keras.models.load_model(hdf_file)
    
    with open(params_file, 'rb') as f:
        params = pickle.load(f)

    data[set_name] = (model.to_json(), model.get_weights(), params)
    print("Done")

print("Writing combined output...")
with open('all_models.pkl', 'wb') as f:
    pickle.dump(data, f)
