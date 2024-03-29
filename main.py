import pygame
import random
import matplotlib.pyplot as plt
import numpy as np
import time 
from methods import Blob, Grid, Soil, compress
from plots import Plots_Blobs

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
energy_to_reproduce = 100
phytogain = 1 
maxage = 300 
gen_var = 0.01 

# Statistics 
time_per_iter_ = []
deaths_stat = [0, 0, 0]
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

else: blobs:list[Blob] = [Blob(random.randint(0, SCREEN_WIDTH // CELL_SIZE-1), random.randint(0, SCREEN_HEIGHT // CELL_SIZE-1)) for _ in range(START_BLOB_NUM)]

# Initialize needed objects
grid = Grid(blobs, SCREEN_WIDTH//CELL_SIZE)
soil = Soil(SCREEN_WIDTH//CELL_SIZE)
p = Plots_Blobs(blobs)

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
                    if not paused: print(iteration_count)
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

    elif keys[pygame.K_h] and keys[pygame.K_UP]: phytogain += 0.002
    elif keys[pygame.K_h] and keys[pygame.K_DOWN]: 
        phytogain -= 0.01
        if phytogain < 0: phytogain=0

    elif keys[pygame.K_g] and keys[pygame.K_UP]: gen_var = np.abs(gen_var+0.002) + 0.001
    elif keys[pygame.K_g] and keys[pygame.K_DOWN]: 
        gen_var -= 0.01
        if gen_var < 0: gen_var=0

    if not paused:
        
        # Let blobs move / Update each blob position
        for blob in blobs:
            blob.move(grid)
        
        # Update the grid wkith the new positions
        grid.update_grid(blobs)

        # Let blobs gain energy form the enviroment 
        for blob in blobs:
            blob.vital(soil, grid, metabolism, phytogain)

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
        blobs = [blob for blob in blobs if blob.is_alive(deaths_stat, soil, maxage)]

            # Refresh the grid to the last update
        grid.update_grid(blobs)

        soil.update_soil()  #Let soil update after being eaten
        soil.difusion()   #Let soil difusion
        soil.update_soil()  #Let soil update after difusioning
            
        # Display iteration's statistics and Store data to create the final graphics
        p.add_blobs(blobs)
        # entropy_stat.append( diversity(blobs) )

        # print(f"Iteration: {iteration_count},  Number of Blobs: {len(blobs)},  ", end='')
        # print(f"Babies: {count_num_baby}, ", end='')
        # print(f"Mean energy: {np.mean([blob.energy for blob in blobs])}, ", end='')
        # print(f"Mean age: {np.mean([blob.age for blob in blobs])}, ", end='')

        # p.show_avg_anatomy()

        # print(f"Conputation time: {time.time()-t_start_iter}, ", end='')
        # print(f"clock_tick set to: {clock_tick}", end='')
        # print()
    

        iteration_count += 1
        time_per_iter_.append(time.time()-t_start_iter)

        # Draw blobs
    screen.fill(BACKGR)
    for i in range(grid.dim):
        for j in range(grid.dim):
            pygame.draw.rect(screen, (0,0,compress(255*soil.get_value(i,j)[0],255)), (i * CELL_SIZE, j * CELL_SIZE, CELL_SIZE, CELL_SIZE))

    for blob in blobs:
        pygame.draw.rect(screen, blob.get_skin(), (blob.x * CELL_SIZE, blob.y * CELL_SIZE, CELL_SIZE, CELL_SIZE))

    pygame.display.flip()
    clock.tick(clock_tick)  # Adjust the speed of the simulation by changing the argument


pygame.quit()


# Show final statistics
fig = plt.figure(figsize=(15,8))

ax0 = fig.add_subplot(2,3,1)
p.plot_population(ax0)

ax1 = fig.add_subplot(2,3,2)
p.plot_error_bar_anatomy(ax1)

ax2 = fig.add_subplot(2,3,3)
ax2.plot([i for i in range(len(time_per_iter_))], time_per_iter_)
ax2.set_xlabel("iteration number")
ax2.set_ylabel("Duration in seg per iteration")

ax3 = fig.add_subplot(2,3,4)
ax3.hist([b.phyto+b.defens-b.offens-b.phago for b in blobs], bins=20)
ax3.set_xlabel("Priority, should be strictly crecent")
ax3.set_ylabel("Absolute frequency")

ax4 = fig.add_subplot(2,3,5)
p.plot_hist2d_reprod(ax4)

ax5 = fig.add_subplot(2, 3, 6)
ax5.plot(deaths_stat, marker='s')
ax5.set_xlabel("0:Depredation, 1:Starvation, 2:Age")
ax5.set_ylabel("Absolute frequency")

# ax7 = fig.add_subplot(2,4,8)
# ax7.plot(range(len(entropy_stat)), entropy_stat, c='g')
# ax7.set_xlabel("time (index)")
# ax7.set_ylabel("Shanon Entropy as Diversity")

plt.show()
