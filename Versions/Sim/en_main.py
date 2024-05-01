import random
import matplotlib.pyplot as plt
import numpy as np
import time 
from en_methods import Blob, Grid


def main(indv_phyto, indv_phago, safe_data_at_file):
    # Parameters of simulation
    SCREEN_WIDTH = 800
    SCREEN_HEIGHT = 800
    CELL_SIZE = 4
    # To initialize blobs
    START_BLOB_NUM = 500
    # Counters
    iteration_count = 0
    count_num_baby = 0

    # Tuneable parameters / characteristics of blobs
    species_info = None
    maxIter = 5e3

    def set_n_species_parameters(n:int):
        species_parameters = []
        for i in range(n):
            num_ind = indv_phago if i else indv_phyto
            species_parameters.append({
                "num_ind": num_ind,
                "phago": 0.05 + 0.9*i/(n-1),
                "phyto": 0.95 - 0.9*i/(n-1),
                "speed": 0.5,
                "vision": 0.6,
                "energy_for_babies": 0.5,
                "number_of_babies": .5,
                "colab": 1/(i+1)
            })

        return species_parameters

    def get_n_species_values(n:int):
        metabolism = 0.5
        energy_to_reproduce = 50
        phytogain = 1.5
        gen_var = 0
        species_info = set_n_species_parameters(n)
        return n, metabolism, energy_to_reproduce, phytogain, gen_var, species_info

    def launch():
        n = 2 
        return get_n_species_values(n)

    def diversity(a:int, b:int) -> float:
        if a == 0: return 0 if b==0 else np.log2(b)
        elif b == 0: return np.log2(a)
        return np.log2(a+b) - (a*np.log2(a)+b*np.log2(b))/(a+b)
            

    num_species, metabolism, energy_to_reproduce, phytogain, gen_var, species_info = launch()
     

    # Statistics
    popu_stat = []
    speed_stat = []
    phyto_stat = []
    phago_stat = []
    vision_stat = []
    phyto_num = []
    phago_num = []
    time_per_iter_ = []
    deaths_stat = [0, 0, 0]
    entropy_stat = []


    blobs = []
    for species in species_info:
        for i in range(species["num_ind"]):
            blobs.append(Blob(x= random.randint(0, SCREEN_WIDTH // CELL_SIZE-1), y=random.randint(0, SCREEN_HEIGHT // CELL_SIZE-1), energy=None, 
            phago=species["phago"], phyto=species["phyto"], 
            speed= species["speed"], vision=species["vision"], 
            energy_for_babies=species["energy_for_babies"], number_of_babies=species["number_of_babies"], 
            colab=species["colab"], skin=None))

    # Initialize needed objects
    grid = Grid(blobs, SCREEN_WIDTH//CELL_SIZE)

    print("""[\033[1;32;40m ECS \033[00m]\033[1;34;40m INFO \033[00m: Showing Manual
        \033[4m ESY interactive chat for Ecosystem-Simulation \033[00m

        \033[1m Pause \033[00m: Press K once to pause the simulation
        \033[1m Speed up \033[00m: Keep pressed L to speedgen_var up the simulation up to minimum execution time.
        \033[1m Slow down \033[00m: Keep pressed J to slow down the simulation.
        \033[1m Stop simulation \033[00m: Press ESC once stop the simulation.
        \033[1m Selection \033[00m: Press two times S to define a Selection Square, which selects the blobs inside to perform an action.
        \033[1m Actions \033[00m: Press from 1 to 5 once to apply a color identifier to selected blobs.
                Press Q once to quit the color identifiers and return to original color code.
                Press E once to eliminate selected blobs
        \033[1m Show features \033[00m: Press A with the mouse in the top of a blob to show that blobs stats and features.
        """)

    while iteration_count < maxIter:
        if not iteration_count % 100 : print(f'{100*iteration_count/maxIter:.2f}%')
        t_start_iter = time.time()

        # Let blobs move / Update each blob position
        for blob in blobs:
            blob.move(grid)
        
        # Update the grid wkith the new positions
        grid.update(blobs)

        # Let blobs gain energy form the enviroment 
        for blob in blobs:
            blob.vital(grid, metabolism, phytogain)

            # Let blobs depredate each other
        for blob in blobs:
            if blob.energy > 0:
                neighbours = grid.blobs_at_tile(blob.x, blob.y)
                for neig in neighbours:
                    blob.fight(neig)

            # Let the blobs feed and check if they may reproduce
        for blob in blobs:
            if blob.energy > 0 and blob.energy >= energy_to_reproduce:
                babies = blob.reproduce(geneticVar=gen_var)
                blobs.extend(babies)
                count_num_baby += len(babies)

            # Remove dead blobs
        blobs = [blob for blob in blobs if blob.is_alive(deaths_stat)]

            # Refresh the grid to the last update
        grid.update(blobs)
            
        # Display iteration's statistics and Store data to create the final graphics
        popu_stat.append(len(blobs))
        act_speed_lst = [blob.speed for blob in blobs]
        act_phyto_lst = [blob.phyto for blob in blobs]
        act_phago_lst = [blob.phago for blob in blobs]
        act_vision_lst = [blob.vision for blob in blobs]
        phyto_stat.append((np.mean(act_phyto_lst), np.std(act_phyto_lst)))
        phago_stat.append((np.mean(act_phago_lst), np.std(act_phago_lst)))
        vision_stat.append((np.mean(act_vision_lst), np.std(act_vision_lst)))

        phgs = sum(blob.phago > .5 for blob in blobs)
        phago_num.append(phgs)

        phys = sum(blob.phyto > .5 for blob in blobs)
        phyto_num.append(phys)

        entropy_stat.append( diversity(phgs, phys) )

        iteration_count += 1
        time_per_iter_.append(time.time()-t_start_iter)


    # Show final statistics
    fig = plt.figure(figsize=(15,8))

    ax1 = fig.add_subplot(2, 3, 1)
    ax1.plot([i for i in range(len(phago_num))], phago_num, color = "red")
    ax1.plot([i for i in range(len(phyto_num))], phyto_num, color= "green")
    ax1.set_xlabel("iteration number")
    ax1.set_ylabel("Populations")

    ax2 = fig.add_subplot(2,3,2)
    ax2.errorbar(x=[i+1 for i in range(len(speed_stat))], y=[avg_std[0] for avg_std in speed_stat],
                yerr=[avg_std[1] for avg_std in speed_stat], fmt='o', linewidth=1, capsize=5, color='orange', 
                errorevery=max(1,len(popu_stat)//25), label = 'speed' )
    ax2.errorbar(x=[i+1 for i in range(len(phyto_stat))], y=[avg_std[0] for avg_std in phyto_stat],
                yerr=[avg_std[1] for avg_std in phyto_stat], fmt='o', linewidth=1, capsize=5, color='green', 
                errorevery=max(1,len(popu_stat)//25), label = 'phyto' )
    ax2.errorbar(x=[i+1 for i in range(len(phago_stat))], y=[avg_std[0] for avg_std in phago_stat],
                yerr=[avg_std[1] for avg_std in phago_stat], fmt='o', linewidth=1, capsize=5, color='red', 
                errorevery=max(1,len(popu_stat)//25), label = 'phago' )
    ax2.errorbar(x=[i+1 for i in range(len(vision_stat))], y=[avg_std[0] for avg_std in vision_stat],
                yerr=[avg_std[1] for avg_std in vision_stat], fmt='o', linewidth=1, capsize=5, color='cyan', 
                errorevery=max(1,len(popu_stat)//25), label = 'vision' )
    ax2.set_xlabel("time (index)")
    ax2.set_ylabel("Averadge stat with std as error bars")
    ax2.legend()

    ax3 = fig.add_subplot(2,3,3)
    ax3.plot([i for i in range(len(time_per_iter_))], time_per_iter_)
    ax3.set_xlabel("iteration number")
    ax3.set_ylabel("Duration in seg per iteration")

    ax4 = fig.add_subplot(2,3,4)
    ax4.plot(phago_num, phyto_num)
    ax4.set_xlabel("phago")
    ax4.set_ylabel("phyto")

    ax5 = fig.add_subplot(2, 3, 5)
    ax5.plot(deaths_stat, marker='s')
    ax5.set_xlabel("0: eaten, 1: starve, 2: old")

    ax6 = fig.add_subplot(2,3,6)
    ax6.plot(range(len(entropy_stat)), entropy_stat, c='g')
    ax6.set_xlabel("time (index)")
    ax6.set_ylabel("Shannon Entropy as diversity")

    if random.random()<.01: plt.show()

    print(phago_num, phyto_num)
    file = open(safe_data_at_file, "a")
    file.write(f'*{phyto_num}{phago_num}')
    print()

if __name__ == '__main__':
    main(int(input()),int(input()),input(f"Enter the COMPLETE NAME of file where the data will append : "))
