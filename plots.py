from methods import Blob
import numpy as np



class Plots_Blobs:
    # anatom_raw  # For each iteration, for each blob, its anatom
    # mean_anatom :# For each iteration, mean of each feature of all blobs
    # std_anatom :    # For each iteration, std of each feature of all blobs

    def __init__(self, blob_array:list[Blob]) -> None:
        self.population = []

        self.anatom_raw = []
        self.num_anato_feat = 0 if not blob_array else len(blob_array[0].anatomy())
        self.mean_anatom = [[] for _ in range(self.num_anato_feat)]
        self.std_anatom = [[] for _ in range(self.num_anato_feat)]

        self.psycho_raw = []
        self.num_psycho_feat = 0 if not blob_array else len(blob_array[0].psycho())
        self.mean_psycho = [[] for _ in range(self.num_anato_feat)]
        self.std_psycho = [[] for _ in range(self.num_anato_feat)]

        self.reprod_raw = []
        self.num_reprod_feat = 0 if not blob_array else len(blob_array[0].reprod())
        self.mean_reprod = [[] for _ in range(self.num_anato_feat)]
        self.std_reprod = [[] for _ in range(self.num_anato_feat)]

        self.num_additions = 0
        self.add_blobs(blob_array)
        
        self.anato_aesthetic_id = {'name':['Phago', 'Phyto', 'Speed', 'Vision', 'Offens', 'Defens'] ,
                                   'col': ['red', 'green', 'orange', 'cyan', 'purple', 'grey']}
        self.psycho_aesthetic_id = {'name':['Curios', 'Agress', 'Colab'] ,
                                   'col': ['pink', 'darkred', 'darkgreen']}
        self.reprod_aesthetic_id = {'name':['energy_for_babies', 'number_of_babies'] ,
                                   'col': ['pink', 'darkblue', 'darkpink']}#Maybe is not needed


    def add_blobs(self, blob_array:list[Blob]) -> None:
        self.population.append(len(blob_array))

        arr = np.array([b.anatomy() for b in blob_array]).T
        self.anatom_raw.append(arr)
        for i,f in enumerate([(np.mean(x),np.std(x)) for x in arr]):
            self.mean_anatom[i].append(f[0])
            self.std_anatom[i].append(f[1])

        arr = np.array([b.psycho() for b in blob_array]).T
        self.psycho_raw.append(arr)
        for i,f in enumerate([(np.mean(x),np.std(x)) for x in arr]):
            self.mean_psycho[i].append(f[0])
            self.std_psycho[i].append(f[1])

        arr = np.array([b.reprod() for b in blob_array]).T
        self.reprod_raw.append(arr)
        for i,f in enumerate([(np.mean(x),np.std(x)) for x in arr]):
            self.mean_reprod[i].append(f[0])
            self.std_reprod[i].append(f[1])

        self.num_additions +=1

    def show_avg_anatomy(self) -> None :
        print(self.num_additions)
        for featu_idx in range(self.num_anato_feat):
            print(f"    {self.anato_aesthetic_id['name']}: mean {self.mean_anatom[featu_idx][-1]}   std {self.std_anatom[featu_idx][-1]}")

    def plot_error_bar_anatomy(self, axes):
        for feature_idx in range(self.num_anato_feat):
            axes.errorbar(x=[i+1 for i in range(self.num_additions)], y=self.mean_anatom[feature_idx],
                yerr=self.std_anatom[feature_idx], fmt='o', linewidth=1, capsize=5, 
                color=self.anato_aesthetic_id['col'][feature_idx], errorevery=max(1,self.num_additions//25), 
                label = self.anato_aesthetic_id['name'][feature_idx] )
        axes.set_xlabel("time (index)")
        axes.set_ylabel("Averadge anatomy stat with std as error bars")
        axes.legend()

    def plot_error_bar_psycho(self, axes):
        for feature_idx in range(self.num_anato_feat):
            axes.errorbar(x=[i+1 for i in range(self.num_additions)], y=self.mean_psycho[feature_idx],
                yerr=self.std_psycho[feature_idx], fmt='o', linewidth=1, capsize=5, 
                color=self.anato_aesthetic_id['col'][feature_idx], errorevery=max(1,self.num_additions//25), 
                label = self.anato_aesthetic_id['name'][feature_idx] )
        axes.set_xlabel("time (index)")
        axes.set_ylabel("Averadge psycho stat with std as error bars")
        axes.legend()

    def plot_hist2d_reprod(self, axes):
        axes.hist2d(self.reprod_raw[-1][0], self.reprod_raw[-1][1], bins=20)
        axes.set_xlabel(self.reprod_aesthetic_id['name'][0])
        axes.set_ylabel(self.reprod_aesthetic_id['name'][1])

    def plot_population(self, axes):
        axes.plot([i+1 for i in range(self.num_additions)], self.population)
        axes.set_xlabel("time (index)")
        axes.set_ylabel("Alive population")
