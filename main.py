import pygame
import random
import matplotlib.pyplot as plt
import numpy as np
import time 
from ECmethods import Blob, Grid
from ECplots import diversity

# Parameters of simulation
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 800
CELL_SIZE = 16
# To initialize blobs
START_BLOB_NUM = 500
PATTERN_BOOL = False
# Counters
iteration_count = 0
count_num_baby = 0
# Colors
BACKGR= (255, 255, 250)

# Tuneable parameters / characteristics of blobs
metabolism = 0.1
energy_to_reproduce = 60
herbogain = 1 
maxage = 300 
gen_var = 0.01 

# Statistics 
popu_stat = []
speed_stat = []
herbo_stat = []
carno_stat = []
vision_stat = []
offens_stat = []
time_per_iter_ = []
deaths_stat = [0, 0, 0]
veloci_stat = []
entropy_stat = []

# Initialize Pygame and set up the screen
pygame.init()
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Blob Simulation")


# Initialize blobs, following the given pattern or not
if PATTERN_BOOL:
    pattern = np.array([[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                        [0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0],
                        [0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0],
                        [1,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                        [1,1,0,0,0,0,0,0,0,0,1,0,0,0,1,0,1,1,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                        [0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                        [0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                        [0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]])   # Define a patern to introduce as initial configuration
    pattern = np.ones((50,50))
    pos = (25,25) # Position to introduce the pattern, coordenates as reference for pattern[0][0]
    blobs = []
    for i in range(len(pattern)):
        for j in range(len(pattern[0])):
            if pattern[i][j] == 1 : blobs.append( Blob( pos[0]+j, pos[1]+i) )

else: blobs = [Blob(random.randint(0, SCREEN_WIDTH // CELL_SIZE-1), random.randint(0, SCREEN_HEIGHT // CELL_SIZE-1)) for _ in range(START_BLOB_NUM)]

# Initialize needed objects
grid = Grid(blobs, SCREEN_WIDTH//CELL_SIZE)

# Main loop
running = True
paused = False
clock = pygame.time.Clock()
clock_tick = 300
select_array = []
last_print_time = time.time()

def be_a_god(Grid, region, action)->None:
    """Execute actions over selected blobs as specified in user manual"""
    pixel_region = [(region[0][0]//CELL_SIZE,region[1][0]//CELL_SIZE), (region[0][1]//CELL_SIZE,region[1][1]//CELL_SIZE)]

    if action in {str(i) for i in range(1,6)}|{'q', 'e'}:
        blob_list = []
        for i in range(min(pixel_region[0]), 1+max(pixel_region[0])):
            for j in range(min(pixel_region[1]), 1+max(pixel_region[1])):
                blob_list.extend(Grid.blobs_at_tile(i,j))

        # Apply color identifier
        if action == '1': 
            for blob in blob_list: blob.skin = (30,30,250)
        elif action == '2': 
            for blob in blob_list: blob.skin = (150,50,200)
        elif action == '3': 
            for blob in blob_list: blob.skin = (100,220,255)
        elif action == '4': 
            for blob in blob_list: blob.skin = (255,30,210)
        elif action == '5': 
            for blob in blob_list: blob.skin = (20,50,100)
        elif action == 'q': 
            for blob in blob_list: blob.skin = None

        # Eliminate
        elif action == 'e': 
            for blob in blob_list: blob.energy = -100

    else: print("[\033[1;32;40m ECS \033[00m]\033[1;31;40m ERROR \033[00m: Given action not executable with selection square")


print("""[\033[1;32;40m ECS \033[00m]\033[1;34;40m INFO \033[00m: Showing Manual
      \033[4m ESY interactive chat for Ecosystem-Simulation \033[00m

      \033[1m Pause \033[00m: Press K once to pause the simulation
      \033[1m Speed up \033[00m: Keep pressed L to speed up the simulation up to minimum execution time.
      \033[1m Slow down \033[00m: Keep pressed J to slow down the simulation.
      \033[1m Stop simulation \033[00m: Press ESC once stop the simulation.
      \033[1m Selection \033[00m: Press two times S to define a Selection Square, which selects the blobs inside to perform an action.
      \033[1m Actions \033[00m: Press from 1 to 5 once to apply a color identifier to selected blobs.
               Press Q once to quit the color identifiers and return to original color code.
               Press E once to eliminate selected blobs
      \033[1m Show features \033[00m: Press A with the mouse in the top of a blob to show that blobs stats and features.
      """)

while running:
    t_start_iter = time.time()
    # TAP JUST ONCE
    for event in pygame.event.get():
        if event.type == pygame.QUIT or blobs==[]:
            running = False
        elif event.type == pygame.KEYDOWN: 
            match event.key :
                case pygame.K_ESCAPE:
                    running = False
                case pygame.K_k:
                    paused = not paused

                case pygame.K_s:
                    if len(select_array) < 2:
                        select_array.append(pygame.mouse.get_pos())
                    else: select_array[1] = pygame.mouse.get_pos()

                case x if x!=pygame.K_s and len(select_array)!=0:
                    if len(select_array) == 2 :
                        be_a_god(grid, select_array, pygame.key.name(event.key))
                        select_array = []
                    elif len(select_array) ==1 :
                        print("[\033[1;32;40m ECS \033[00m]\033[1;31;40m ERROR \033[00m: Selection square is not defind to execute an action")
                        select_array = []

                case pygame.K_a:
                    mouse_pos = pygame.mouse.get_pos()
                    blobs_at_tile = grid.blobs_at_tile(mouse_pos[0]//CELL_SIZE, mouse_pos[1]//CELL_SIZE)
                    if blobs_at_tile:
                        print(f"[\033[1;32;40m ECS \033[00m]\033[1;34;40m INFO \033[00m: Showing features of {blobs_at_tile[0]}")
                        blobs_at_tile[0].show_features()
                    else: print("[\033[1;32;40m ECS \033[00m]\033[1;31;40m ERROR \033[00m: No blob identified ")
                

    # KEEP PRESSING
    keys = pygame.key.get_pressed()
    if keys[pygame.K_l]: clock_tick +=2
    elif keys[pygame.K_j]: 
        clock_tick -=2   
        if clock_tick < 0: clock_tick=1

    elif keys[pygame.K_m] and keys[pygame.K_UP]: metabolism += 0.01
    elif keys[pygame.K_m] and keys[pygame.K_DOWN]: 
        metabolism -= 0.01
        if metabolism < 0: metabolism=0

    elif keys[pygame.K_h] and keys[pygame.K_UP]: herbogain += 0.002
    elif keys[pygame.K_h] and keys[pygame.K_DOWN]: 
        herbogain -= 0.01
        if herbogain < 0: herbogain=0

    elif keys[pygame.K_g] and keys[pygame.K_UP]: gen_var += 0.002
    elif keys[pygame.K_g] and keys[pygame.K_DOWN]: 
        gen_var -= 0.01
        if gen_var < 0: gen_var=0

    if not paused:
        
        # Let blobs move / Update each blob position
        for blob in blobs:
            blob.move(grid)
        
        # Update the grid with the new positions
        grid.update(blobs)

        # Let blobs gain energy form the enviroment 
        for blob in blobs:
            blob.vital(grid, metabolism, herbogain)

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
        blobs = [blob for blob in blobs if blob.is_alive(deaths_stat, maxage)]

            # Refresh the grid to the last update
        grid.update(blobs)
            
        # Display iteration's statistics and Store data to create the final graphics
        popu_stat.append(len(blobs))
        act_speed_lst = [blob.speed for blob in blobs]
        act_herbo_lst = [blob.herbo for blob in blobs]
        act_carno_lst = [blob.carno for blob in blobs]
        act_vision_lst = [blob.vision for blob in blobs]
        act_offens_lst = [blob.offens for blob in blobs]
        speed_stat.append((np.mean(act_speed_lst), np.std(act_speed_lst)))
        herbo_stat.append((np.mean(act_herbo_lst), np.std(act_herbo_lst)))
        carno_stat.append((np.mean(act_carno_lst), np.std(act_carno_lst)))
        vision_stat.append((np.mean(act_vision_lst), np.std(act_vision_lst)))
        offens_stat.append((np.mean(act_offens_lst), np.std(act_offens_lst)))
        if len(popu_stat)!=1: veloci_stat.append( (popu_stat[-1]-popu_stat[-2])/popu_stat[-1] )
        entropy_stat.append( diversity(blobs) )

        # print(f"Iteration: {iteration_count},  Number of Blobs: {len(blobs)},  ", end='')
        # print(f"Babies: {count_num_baby}, ", end='')
        # print(f"Mean energy: {np.mean([blob.energy for blob in blobs])}, ", end='')
        # print(f"Mean age: {np.mean([blob.age for blob in blobs])}, ", end='')
        # print(f"Mean speed: {np.mean(act_speed_lst)},  ", end='')
        # print(f"Mean herbiborous: {np.mean(act_herbo_lst)}, ", end='')
        # print(f"Mean carnivorous: {np.mean(act_carno_lst)}, ", end='')
        # print(f"Mean vision: {np.mean(act_vision_lst)}, ", end='')
        # print(f"Mean offens: {np.mean(act_offens_lst)}, ", end='')
        # print(f"Conputation time: {time.time()-t_start_iter}, ", end='')
        # print(f"clock_tick set to: {clock_tick}", end='')
        # print()
    

        iteration_count += 1
        time_per_iter_.append(time.time()-t_start_iter)

        # Draw blobs
    screen.fill(BACKGR)
    for blob in blobs:
        pygame.draw.rect(screen, blob.get_skin(), (blob.x * CELL_SIZE, blob.y * CELL_SIZE, CELL_SIZE, CELL_SIZE))

    pygame.display.flip()
    clock.tick(clock_tick)  # Adjust the speed of the simulation by changing the argument


pygame.quit()



# Show final statistics
fig = plt.figure(figsize=(15,8))

ax0 = fig.add_subplot(2,4,1)
ax0.plot([i+1 for i in range(len(popu_stat))], popu_stat)
ax0.set_xlabel("time (index)")
ax0.set_ylabel("Alive population")

ax1 = fig.add_subplot(2,4,2)
ax1.errorbar(x=[i+1 for i in range(len(speed_stat))], y=[avg_std[0] for avg_std in speed_stat],
              yerr=[avg_std[1] for avg_std in speed_stat], fmt='o', linewidth=1, capsize=5, color='orange', 
              errorevery=max(1,len(popu_stat)//25), label = 'speed' )
ax1.errorbar(x=[i+1 for i in range(len(herbo_stat))], y=[avg_std[0] for avg_std in herbo_stat],
              yerr=[avg_std[1] for avg_std in herbo_stat], fmt='o', linewidth=1, capsize=5, color='green', 
              errorevery=max(1,len(popu_stat)//25), label = 'herbo' )
ax1.errorbar(x=[i+1 for i in range(len(carno_stat))], y=[avg_std[0] for avg_std in carno_stat],
              yerr=[avg_std[1] for avg_std in carno_stat], fmt='o', linewidth=1, capsize=5, color='red', 
              errorevery=max(1,len(popu_stat)//25), label = 'carno' )
ax1.errorbar(x=[i+1 for i in range(len(vision_stat))], y=[avg_std[0] for avg_std in vision_stat],
              yerr=[avg_std[1] for avg_std in vision_stat], fmt='o', linewidth=1, capsize=5, color='cyan', 
              errorevery=max(1,len(popu_stat)//25), label = 'vision' )
ax1.errorbar(x=[i+1 for i in range(len(offens_stat))], y=[avg_std[0] for avg_std in offens_stat],
              yerr=[avg_std[1] for avg_std in offens_stat], fmt='o', linewidth=1, capsize=5, color='purple', 
              errorevery=max(1,len(popu_stat)//25), label = 'offens' )
ax1.set_xlabel("time (index)")
ax1.set_ylabel("Averadge stat with std as error bars")
ax1.legend()

ax2 = fig.add_subplot(2,4,3)
ax2.plot([i for i in range(len(time_per_iter_))], time_per_iter_)
ax2.set_xlabel("iteration number")
ax2.set_ylabel("Duration in seg per iteration")

ax3 = fig.add_subplot(2,4,4, projection='3d')
ax3.scatter([blob.offens for blob in blobs], [blob.defens for blob in blobs], [blob.carno for blob in blobs], c=[blob.herbo for blob in blobs], s=5, alpha=0.5)
# ax3.scatter([blob.fav_meal[4] for blob in blobs], [blob.fav_meal[5] for blob in blobs], [blob.fav_meal[2] for blob in blobs], c='red', s=5, alpha=0.5)
ax3.set_xlim(0,1)
ax3.set_ylim(0,1)
ax3.set_zlim(0,1)
ax3.set_xlabel("offens")
ax3.set_ylabel("defens")
ax3.set_zlabel("carno")
# ax3.quiver([blob.carno for blob in blobs], [blob.vision for blob in blobs], [blob.speed for blob in blobs], [blob.fav_meal[0] for blob in blobs], [blob.fav_meal[3] for blob in blobs], [blob.fav_meal[2] for blob in blobs], length=0.1, normalize=True)

ax4 = fig.add_subplot(2,4,5, projection='3d')
ax4.plot([avg_std[0] for avg_std in herbo_stat], [avg_std[0] for avg_std in carno_stat], [avg_std[0] for avg_std in offens_stat])
ax4.set_xlabel("herbo")
ax4.set_ylabel("carno")
ax4.set_zlabel("offens")

ax5 = fig.add_subplot(2,4,6)
ax5.hist2d([blob.number_of_babies for blob in blobs], [blob.energy_for_babies for blob in blobs], bins=20)
ax5.set_xlabel("number_of_babies")
ax5.set_ylabel("energy_for_babies")

ax6 = fig.add_subplot(2, 4, 7)
ax6.plot(deaths_stat, marker='s')

ax7 = fig.add_subplot(2,4,8)
ax7.plot(range(len(entropy_stat)), entropy_stat, c='g')
ax7.set_xlabel("time (index)")
ax7.set_ylabel("Specific velocity of growth")

plt.show()
