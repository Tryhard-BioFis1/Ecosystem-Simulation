import pygame
import random
import matplotlib.pyplot as plt
import numpy as np
import time 

def noise(sigma:float=0.02)->float:
    """Encapsulates the generation of a random number with triangular distribution"""
    return np.random.normal(scale=sigma)

def compress(a:float, max_lim:float=1)->float:
    """
    Given a float 'a', if overpass the limits 0 (lower limit) or 'max_lim' 
    return the overpassed limit, else return 'a'
    """
    if a>max_lim: return max_lim
    if a<0: return 0
    return a

def mod_dist(a:int, b:int, n:int)->int:
    """Calculates the minimun SIGNED steps to get from a to b modulo n"""
    a_ , b_ = a%n , b%n
    dst = min(abs(b_-a_), n-abs(b_-a_))
    if (a_+dst)%n == b_ : return dst
    return -dst


class Grid:
    """Consists in the grid where blobs move and interact"""
    dim : int   # lenght of grids side -> Grid has dim*dim shape
    dic : dict  # dictionary which stores the blobs, keys are tuple (x,y) of non empty tiles
                #   and values list with those blobs which are at that position

    def __init__(self, blob_list:list['Blob'], dim:int)->None:
        self.dic = {}
        self.dim = dim
        for blob in blob_list:
            if (blob.x, blob.y) in self.dic:
                self.dic[(blob.x, blob.y)].append(blob)
            else: 
                self.dic[(blob.x, blob.y)] = [blob]

    def blobs_at_tile(self, c_x:int, c_y:int)->list['Grid']:
        """Returns list with blobs at given tile"""
        if (c_x, c_y) not in self.dic: return []
        else: return self.dic[(c_x,c_y)]

    def get_neighbours_dist(self, c_x:int, c_y:int, vrad:int)->list['Blob']:
        """Returns the neighbours of tile (c_x,c_y) at certain distance of vrad"""
        neighbours = []
        if vrad == 0: return self.dic[(c_x,c_y)]

        for j in range(c_y - vrad, c_y + 1 + vrad):
            neighbours.extend(self.blobs_at_tile( (c_x - vrad)%self.dim, j%self.dim ))
            neighbours.extend(self.blobs_at_tile( (c_x + vrad)%self.dim, j%self.dim ))

        for i in range(c_x - vrad + 1, c_x + vrad):
            neighbours.extend(self.blobs_at_tile( (i)%self.dim, (c_y + vrad)%self.dim ))
            neighbours.extend(self.blobs_at_tile( (i)%self.dim, (c_y - vrad)%self.dim ))

        return neighbours
    
    def get_neighbours_inter(self, c_x:int, c_y:int, vrad_min:int, vrad_max:int)->list['Blob']:
        """Returns the neighbours of tile (c_x,c_y) at certain interval distance from vrad_min to vrad_max"""
        neighbours = []
        for vrad_i in range(vrad_min, vrad_max+1):
            neighbours.extend(self.get_neighbours_dist(c_x, c_y, vrad_i))
        return neighbours

    def update(self, blob_list:list['Blob'])->None:
        """Update sel.dic because the blobs have moved"""
        self.dic = {}
        for blob in blob_list:
            if (blob.x, blob.y) in self.dic:
                self.dic[(blob.x, blob.y)].append(blob)
            else: 
                self.dic[(blob.x, blob.y)] = [blob]


class Blob:
    x : int         # horizontal position 
    y : int         # vertical position
    energy : int    # 0-100 amount of energy stored 
    carno : float   # 0-1 parameter how many energy gets from feeding from preys
    herbo : float   # 0-1 parameter how many energy gets from the surroundings
    speed : float   # 0-1 parameter in averadge how many tiles moves per iteration
    vision : float  # 0-1 parameter divided by 0.15 is the how far it sees
    age: int        # ** parameter the current age of the blob 
    offens : float  
    energy_for_babies: float # > 1 parameter that indicates the number of babies
    number_of_babies: float # 0-1 parameter that establishes the probability of moving randomly in case there are no reasons for doing it
    curiosity: float  # 0-1 parameter that establishes the probability of moving randomly in case there are no reasons for doing it
    collab: float #0-1 parameter that establishes how likely will share its energy with other blobs with similar stats


    def __init__(self, x=None, y=None, energy=None, carno=None, herbo=None, speed=None, age=None, vision=None, offens=None, 
                 energy_for_babies=None, number_of_babies=None, curiosity=None, agressive=None, colab=None)->None:
            
            # Each variable has a predeterminate value unless one is specified 
            self.x = 0 if x is None else x
            self.y = 0 if y is None else y
            self.energy = random.randint(20,80) if energy is None else energy
            self.carno= random.uniform(0.1, 0.9) if carno is None else carno
            self.herbo = random.uniform(0.1, 0.9) if herbo is None else herbo
            self.speed = random.uniform(0.1, 0.9) if speed is None else speed
            self.age = random.randint(1,500) if age is None else age
            self.vision = random.uniform(0.1, 0.9) if vision is None else vision
            self.offens = random.uniform(0.1, 0.9) if offens is None else offens
            self.energy_for_babies = random.uniform(0.1, 0.9) if energy_for_babies is None else energy_for_babies
            self.number_of_babies = random.uniform(0.1, 0.9) if number_of_babies is None else number_of_babies
            self.curiosity = random.uniform(0, 1) if curiosity is None else curiosity
            self.agressive = random.uniform(0, 1) if agressive is None else agressive
            self.colab = random.uniform(0, 1) if colab is None else colab

    def compute_next_move(self, grid:'Grid')->tuple[int,int]: 
        """Compute a factor which determines how Self will move"""
        dx_prop , dy_prop = 0,0
        for k in range(1, int(1 + self.vision//0.2)):
            for blobi in grid.get_neighbours_dist(self.x, self.y, k):
                if self.offens > 1.2*blobi.offens: #Has to be modified in order to establish a more random interaction
                    dx_prop += 2/3*mod_dist(self.x, blobi.x, grid.dim)/k
                    dy_prop += 2/3*mod_dist(self.y, blobi.y, grid.dim)/k
                elif blobi.offens > 1.2*self.offens:
                    dx_prop -= 2/3*mod_dist(self.x, blobi.x, grid.dim)/k
                    dy_prop -= 2/3*mod_dist(self.y, blobi.y, grid.dim)/k
                else:
                    dx_prop -= 1/3*blobi.herbo*mod_dist(self.x,blobi.x,grid.dim)/k
                    dy_prop -= 1/3*blobi.herbo*mod_dist(self.y,blobi.y,grid.dim)/k

        abs_prop = np.sqrt(dx_prop*dx_prop+dy_prop*dy_prop) 

        dx, dy = 0, 0
        if abs_prop < 1e-6:
            if random.random() < self.curiosity:
                dx = random.randint(-1, 1)
                dy = random.randint(-1, 1)
        else:
            if random.random() < abs(dx_prop/abs_prop):
                dx = int(np.sign(dx_prop))
            if random.random() < abs(dy_prop/abs_prop):
                dy = int(np.sign(dy_prop))

        return dx, dy

    def move(self, grid, movement_energy=0.5)->None:
        """Update the position of Self"""
        if random.random() < self.speed:  #creo que hay un error en como estamos entendiendo speed. A estas alturas, me parece que los blobs 
                                        #deberían elegir si moverse o no. Hay posiciones en las que evidentemente sale más rentable no moverse. Por qué lo harían aún siendo capaces de hacerlo

            dx , dy = self.compute_next_move(grid)

            # Update position with periodic boundary conditions
            if dx != 0 or dy != 0:
                self.x = (self.x + dx) % (SCREEN_WIDTH // CELL_SIZE)
                self.y = (self.y + dy) % (SCREEN_HEIGHT // CELL_SIZE)
                self.energy -= movement_energy

    def fight(self, blob:'Blob', easy_eat:float =0.25)->None:
        """Self and a Blob fight and the result if updated"""
        if self.energy>0 and blob.energy>0: # check if both are alive
            if self.offens > blob.offens:       #self eats blob 
                if self.offens-blob.offens > easy_eat or random.random() < (1/easy_eat)*(self.offens-blob.offens):
                        self.energy += blob.energy * self.carno 
                        blob.energy = -100
            else:       # blob eats self 
                if self.offens-blob.offens < -easy_eat or random.random() < -(1/easy_eat)*(self.offens-blob.offens):
                        blob.energy += self.energy * blob.carno 
                        self.energy = -100



    def update_vital(self, grid:'Grid', metabo:float = 0.1)->None:
        """
            Blob gains energy from surrondings with probability based on herbo level
            If multiple blobs are in the same tile the energy is distributed among them
        """
        if random.random() < self.herbo/(len(grid.blobs_at_tile(self.x, self.y))+0.5*len(grid.get_neighbours_dist(self.x, self.y, 1)) 
                                         +0.25*len(grid.get_neighbours_dist(self.x, self.y, 2)) + 0.0*len(grid.get_neighbours_dist(self.x, self.y, 3))):
            self.energy += 0.7
        self.age +=1
        
        self.energy -= metabo*(1+ (self.herbo+self.carno+self.speed+self.vision+self.offens)/5 )
        # print(self.herbo+self.carno+self.speed+self.vision+self.offens)

    def is_alive(self)->None:
        """Checks if blob remains alive or is already dead"""
        return self.energy > 0 and self.age < 500
    
    def reproduce(self, giving_birth_cost=1.2)->list['Blob']:
        """If blob pass a energy threshold, a new Blob is born with some mutations"""
        babies_energy = self.energy * self.energy_for_babies
        self.energy -= babies_energy*giving_birth_cost
        babies = []
        k = int(self.number_of_babies//0.15)
        for _ in range(k):
            babies.append(Blob(self.x+random.randint(-1, 1), self.y+random.randint(-1, 1), energy= babies_energy/k, carno=compress(self.carno+noise()), 
                    herbo=compress(self.herbo+noise()), speed=compress(self.speed+noise()), age=1, 
                    vision=compress(self.vision + noise()), offens=compress(self.offens + noise()), 
                    energy_for_babies=compress(self.energy_for_babies+noise()), number_of_babies=compress(self.number_of_babies+noise()), 
                    curiosity=compress(self.curiosity+noise()), agressive=compress(self.agressive+noise()), colab=compress(self.colab+noise())))

        return babies



# Parameters of simulation
SCREEN_WIDTH = 500
SCREEN_HEIGHT = 500
CELL_SIZE = 5
# To initialize blobs
START_BLOB_NUM = 500
PATTERN_BOOL = False
# Counters
iteration_count = 0
count_num_baby = 0
# Colors
BACKGR= (255, 255, 250)
GREEN = (0, 255, 0)

# Tuneable parameters / characteristics of blobs
metabolism = 0.1
energy_to_reproduce = 60

# Statistics 
popu_stat = []
speed_stat = []
herbo_stat = []
carno_stat = []
vision_stat = []
offens_stat = []
time_per_iter_ = []

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

else: blobs = [Blob(random.randint(0, SCREEN_WIDTH // CELL_SIZE), random.randint(0, SCREEN_HEIGHT // CELL_SIZE)) for _ in range(START_BLOB_NUM)]

# Initialize needed objects
grid = Grid(blobs, SCREEN_WIDTH//CELL_SIZE)

# Main loop
running = True
paused = False
clock = pygame.time.Clock()
clock_tick = 500

while running:
    t_start_iter = time.time()
    for event in pygame.event.get():
        if event.type == pygame.QUIT or blobs==[]:
            running = False

    keys = pygame.key.get_pressed()
    if keys[pygame.K_l]: clock_tick +=2
    elif keys[pygame.K_j]: clock_tick -=2

    # Refresh the grid to the last update
    grid.update(blobs)

    # Let blobs move / Update each blob position
    for blob in blobs:
        blob.move(grid)
    
    # Update the grid with the new positions
    grid.update(blobs)

    # Let blobs gain energy form the enviroment 
    time_vital = time.time()
    for blob in blobs:
        blob.update_vital(grid, metabolism)
    # print("Time_vital: ", time.time() - time_vital)

    time_carno = time.time()
    # Let blobs depredate each other
    for blob in blobs:
        if blob.energy > 0:
            neighbours = grid.blobs_at_tile(blob.x, blob.y)

            for neig in neighbours:
                blob.fight(neig)
    # print("Time carno: ", time.time() - time_carno)

    time_repro = time.time()
    # Let the blobs feed and check if they may reproduce
    for blob in blobs:
        if blob.energy > 0 and blob.energy >= energy_to_reproduce:
            babies = blob.reproduce()
            blobs.extend(babies)
            count_num_baby += len(babies)
    # print("Time rpro: ", time.time() - time_repro)

    time_remove = time.time()
    # Remove dead blobs
    blobs = [blob for blob in blobs if blob.is_alive()]
    # print("Time_remove: ", time.time() - time_remove)

    # Draw blobs
    screen.fill(BACKGR)
    for blob in blobs:
        pygame.draw.rect(screen, (compress(255*blob.carno,255), compress(255*blob.herbo,255), 0), (blob.x * CELL_SIZE, blob.y * CELL_SIZE, CELL_SIZE, CELL_SIZE))
        
    pygame.display.flip()
    clock.tick(clock_tick)  # Adjust the speed of the simulation by changing the argument

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
    # print(np.mean([blob.offens for blob in blobs]), end='')
    print(f"clock_tick set to: {clock_tick}", end='')
    print()
    

    iteration_count += 1
    time_per_iter_.append(time.time()-t_start_iter)

pygame.quit()



# Show final statistics
fig = plt.figure(figsize=(15,8))

ax0 = fig.add_subplot(2,3,1)
ax0.plot([i+1 for i in range(len(popu_stat))], popu_stat)
ax0.set_xlabel("time (index)")
ax0.set_ylabel("Alive population")

ax1 = fig.add_subplot(2,3,2)
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

ax2 = fig.add_subplot(2,3,3)
ax2.plot([i for i in range(len(time_per_iter_))], time_per_iter_)
ax2.set_xlabel("iteration number")
ax2.set_ylabel("Duration in seg per iteration")

ax3 = fig.add_subplot(2,3,4, projection='3d')
ax3.scatter([blob.carno for blob in blobs], [blob.vision for blob in blobs], [blob.speed for blob in blobs], c=[blob.offens for blob in blobs], s=5, alpha=0.5)
ax3.set_xlim(0,1)
ax3.set_ylim(0,1)
ax3.set_zlim(0,1)
ax3.set_xlabel("carno")
ax3.set_ylabel("vision")
ax3.set_zlabel("speed")

ax4 = fig.add_subplot(2,3,5, projection='3d')
ax4.plot([avg_std[0] for avg_std in herbo_stat], [avg_std[0] for avg_std in carno_stat], [avg_std[0] for avg_std in offens_stat])
ax4.set_xlabel("herbo")
ax4.set_ylabel("carno")
ax4.set_zlabel("offens")

ax5 = fig.add_subplot(2,3,6)
ax5.hist2d([blob.number_of_babies for blob in blobs], [blob.energy_for_babies for blob in blobs], bins=20)
ax5.set_xlabel("number_of_babies")
ax5.set_ylabel("energy_for_babies")

plt.show()
