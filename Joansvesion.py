import pygame
import random
import matplotlib.pyplot as plt
import numpy as np
import time



# Define functions and objects
def noise()->float:
    """Encapsulates the generation of a random number with triangular distribution"""
    return random.triangular(-mutation,mutation)

def compress(a:float, max_lim:float =1)->float:
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
    dic : dict  # dictionaty which stores the blobs, keys are tuple (x,y) of non empty tiles
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
        """
            Returns list with blobs at given tile
        """
        if (c_x, c_y) not in self.dic: return []
        else: return self.dic[(c_x,c_y)]

    def get_neighbours_dist(self, c_x:int, c_y:int, vrad:int)->list['Blob']:
        """Returns the neighbours of tile (c_x,c_y) at certain distance of vrad
        """
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
        """Returns the neighbours of tile (c_x,c_y) at certain 
            interval distance from vrad_min to vrad_max"""
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
    energy : float    # 0-100 amount of energy stored 
    carbo : float   # 0-1 parameter how many energy gets from feeding from preys
    herbo : float   # 0-1 parameter how many energy gets from the surroundings
    speed : float   # 0-1 parameter in averadge how many tiles moves per iteration
    vision : float  # 0-1 parameter divided by 0.15 is the how far it sees
    age: int        # ** parameter the current age of the blob
    energy_for_babies : float # 0-1 parameter that indicates the fraction of energy transmited to babies when reproduce
    number_of_babies : float # > 1 parameter that indicates the number of babies
    curiosity : float # 0-1 parameter that establishes the probability of moving randomly in case there are no reasons for doing it
    agressive : float #0-1 parameter that establishes the risk an individual is able to take

    def __init__(self, x=None, y=None, energy=None, carno=None, herbo=None, speed=None, age=None, vision=None, energy_for_babies=None, number_of_babies=None, curiosity= None, agressive = None)->None:
            # Each variable has a predeterminate value unless one is specified 
            self.x = 0 if x is None else x #+ random.randint(-1, 1)
            self.y = 0 if y is None else y #+ random.randint(-1, 1)
            self.energy = random.randint(20,80) if energy is None else energy
            self.herbo = (random.uniform(0, 1))if herbo is None else herbo
            self.carno= (1-(self.herbo)**exp)**(1/exp) if carno is None else carno
            self.speed = random.uniform(0.25, 0.75) if speed is None else speed
            self.age = random.randint(1, maxage) if age is None else age
            self.vision = random.uniform(0.1, 0.9) if vision is None else vision
            self.energy_for_babies = random.uniform(0.1, 0.9) if energy_for_babies is None else energy_for_babies
            self.number_of_babies = random.uniform(0, 1) if number_of_babies is None else number_of_babies
            self.curiosity = random.uniform(0, 1) if curiosity is None else curiosity
            self.agressive = random.uniform(0, 1) if agressive is None else agressive
    
    def compute_next_move(self, blob: 'Blob'): 
        """Compute a factor which determines how Self would move"""
        dx_prop , dy_prop = 0,0
        for k in range(1, int(1 + self.vision//0.2)):
            for blobi in grid.get_neighbours_dist(self.x, self.y, k):
                if self.carno > possibleeat*blobi.carno:  #possibleeat parameter will be deleted in the near future
                    dx_prop += 2/3*mod_dist(self.x, blobi.x, grid.dim)/k
                    dy_prop += 2/3*mod_dist(self.y, blobi.y, grid.dim)/k
                elif blobi.carno >possibleeat*self.carno:
                    dx_prop -= 2/3*mod_dist(self.x, blobi.x, grid.dim)/k
                    dy_prop -= 2/3*mod_dist(self.y, blobi.y, grid.dim)/k
                else:
                    dx_prop -= 1/3*blobi.herbo*mod_dist(self.x,blobi.x,grid.dim)/k
                    dy_prop -= 1/3*blobi.herbo*mod_dist(self.y,blobi.y,grid.dim)/k

        abs_prop = np.sqrt(dx_prop*dx_prop+dy_prop*dy_prop) 

        dx , dy = 0 , 0
        if abs_prop < 1e-6 : 
            if random.random() < self.curiosity:  
                dx = random.randint(-1, 1)
                dy = random.randint(-1, 1)
        else:
            if random.random() < abs(dx_prop/abs_prop):
                dx = int(np.sign(dx_prop))
            if random.random() < abs(dy_prop/abs_prop): #aquí antes había un elif, por qué?? En teoría los Blobs se pueden mover en diagonal
                dy = int(np.sign(dy_prop))

        return dx, dy

    def move(self, grid):
        if random.random() < self.speed:  #creo que hay un error en como estamos entendiendo speed. A estas alturas, me parece que los blobs 
                                        #deberían elegir si moverse o no. Hay posiciones en las que evidentemente sale más rentable no moverse. Por qué lo harían aún siendo capaces de hacerlo

            dx , dy = self.compute_next_move(grid)

            # Update position with periodic boundary conditions
            if dx != 0 or dy != 0:
                self.x = (self.x + dx) % (SCREEN_WIDTH // CELL_SIZE)
                self.y = (self.y + dy) % (SCREEN_HEIGHT // CELL_SIZE)
                self.energy -= movement_energy

    def update_vital(self, grid:'Grid', metabo:float = 0.1)->None:
        """
            Blob gains energy from surrondings with probability based on herbo level
            If multiple blobs are in the same tile the energy is distributed among them
        """
        if random.random() < self.herbo/(len(grid.blobs_at_tile(self.x, self.y))+
                                         0.5*len(grid.get_neighbours_dist(self.x, self.y, 1))+0.25*len(grid.get_neighbours_dist(self.x, self.y, 2)) + 0.125*len(grid.get_neighbours_dist(self.x, self.y, 3))):
            self.energy += 1
        self.age +=1
        self.energy -=metabo

    def is_alive(self)->None:
        """Checks if blob remains alive or is already dead"""
        return self.energy > 0 and self.age < maxage
    
    def reproduce(self)->None:#'Blob'|None
        """If blob pass a energy threshold, a new Blob is born with some mutations"""
        babies_energy = self.energy * self.energy_for_babies
        self.energy -= babies_energy*giving_birth_cost
        babies = []
        k = int(self.number_of_babies//0.15)
        for _ in range(k):
            noise_for_feeding = noise()
            babies.append(Blob(self.x + random.randint(-1, 1), self.y + random.randint(-1, 1), babies_energy/k, (1-(compress(self.herbo+noise_for_feeding))**exp)**(1/exp), compress(self.herbo+noise_for_feeding),
                        compress(self.speed+noise()), age=1, vision =  compress(self.vision + noise()), energy_for_babies= compress(self.energy_for_babies + noise()), 
                        number_of_babies= compress(self.number_of_babies + noise()), curiosity= compress(self.curiosity + noise()), agressive= compress(self.agressive + noise())))
        
        return babies


exp = 1
# Parameters of simulation
SCREEN_WIDTH = 200
SCREEN_HEIGHT = 200
CELL_SIZE = 5

# To initialize blobs
START_BLOB_NUM = 100
PATTERN_BOOL = False
# Counters
iteration_count = 0
count_num_baby = 0
# Colors
BACKGR= (255, 255, 250)
GREEN = (0, 255, 0)

# Tuneable parameters / characteristics of blobs
metabolism = 0.2
possibleeat = 1.2
maxage = 1500
mutation = 0.05
movement_energy = 0.5
energy_to_reproduce = 60
giving_birth_cost = 1.2
trophic_chain_loss = 0.3
# Statistics 
popu_stat = []
speed_stat = []
herbo_stat = []
carno_stat = []
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

# Main loop
running = True
clock = pygame.time.Clock()

while running:
    t_start_iter = time.time()
    for event in pygame.event.get():
        if event.type == pygame.QUIT or blobs==[]:
            running = False

    # Initialize needed objects
    grid = Grid(blobs, SCREEN_WIDTH//CELL_SIZE)

    # Let blobs move / Update each blob position
    for blob in blobs:
        blob.move(grid) #grid   MAYBE A GREEDY IS IMPLEMENTED IN FUNCTION OF grid 
    
    # Update the grid with the new positions
    grid.update(blobs)

    for blob in blobs:
        blob.update_vital(grid, metabolism)

    # Carno
    for blob in blobs:
        if blob.energy > 0:
            neighbours = grid.blobs_at_tile(blob.x, blob.y)

            for neig in neighbours:
                if blob.carno > possibleeat*neig.carno: 
                    blob.energy += min(trophic_chain_loss*neig.energy, 10) 
                    neig.energy = -1

    # Let the blobs feed and check if they may reproduce
    for blob in blobs:
        if blob.energy > 0 and blob.energy >= energy_to_reproduce:
            babies = blob.reproduce()
            blobs.extend(babies)
            count_num_baby += len(babies)

    # Remove dead blobs
    blobs = [blob for blob in blobs if blob.is_alive()]

    # Draw blobs
    screen.fill(BACKGR)
    for blob in blobs:
        pygame.draw.rect(screen, (compress(255*blob.carno,255), compress(255*blob.herbo,255), 0), (blob.x * CELL_SIZE, blob.y * CELL_SIZE, CELL_SIZE, CELL_SIZE))
        
    pygame.display.flip()
    clock.tick(500)  # Adjust the speed of the simulation by changing the argument

    # Display iteration's statistics and Store data to create the final graphics
    popu_stat.append(len(blobs))
    act_speed_lst = [blob.speed for blob in blobs]
    act_herbo_lst = [blob.herbo for blob in blobs]
    act_carno_lst = [blob.carno for blob in blobs]
    speed_stat.append((np.mean(act_speed_lst), np.std(act_speed_lst)))
    herbo_stat.append((np.mean(act_herbo_lst), np.std(act_herbo_lst)))
    carno_stat.append((np.mean(act_carno_lst), np.std(act_carno_lst)))
    
    print(f"Iteration: {iteration_count},  Number of Blobs: {len(blobs)},  ", end='')
    print(f"Babys: {count_num_baby}, ", end='')
    # print(f"Mean energy: {np.mean([blob.energy for blob in blobs])}, ", end='')
    # print(f"Mean age: {np.mean([blob.age for blob in blobs])}, ", end='')
    # print(f"Mean speed: {np.mean(act_speed_lst)},  ", end='')
    # print(f"Mean herbiborous: {np.mean(act_herbo_lst)}, ", end='')
    print(f"Mean carnivorous: {np.mean(act_carno_lst)}, ", end='')
    # print(f"Conputation time: {time.time()-t_start_iter}, ", end='')
    print()

    iteration_count += 1
    total = time.time()-t_start_iter
    time_per_iter_.append(time.time()-t_start_iter)

pygame.quit()



# Show final statistics
fig = plt.figure(figsize=(12,8))
# fig, axes = plt.subplots(2,3, figsize=(12,8))

ax0 = fig.add_subplot(2,3,1)
ax0.plot([i+1 for i in range(len(popu_stat))], popu_stat)
ax0.set_xlabel("time (index)")
ax0.set_ylabel("Alive population")

ax1 = fig.add_subplot(2,3,2)
ax1.errorbar(x=[i+1 for i in range(len(speed_stat))], y=[avg_std[0] for avg_std in speed_stat],
              yerr=[avg_std[1] for avg_std in speed_stat], fmt='o', linewidth=2, capsize=6, color='orange', 
              errorevery=max(1,len(popu_stat)//25), label = 'speed'  )
ax1.errorbar(x=[i+1 for i in range(len(herbo_stat))], y=[avg_std[0] for avg_std in herbo_stat],
              yerr=[avg_std[1] for avg_std in herbo_stat], fmt='o', linewidth=2, capsize=6, color='green', 
              errorevery=max(1,len(popu_stat)//25), label = 'herbo' )
ax1.errorbar(x=[i+1 for i in range(len(carno_stat))], y=[avg_std[0] for avg_std in carno_stat],
              yerr=[avg_std[1] for avg_std in carno_stat], fmt='o', linewidth=2, capsize=6, color='red', 
              errorevery=max(1,len(popu_stat)//25), label = 'carno' )
ax1.set_xlabel("time (index)")
ax1.set_ylabel("Averadge stat with std as error bars")
ax1.legend()

ax2 = fig.add_subplot(2,3,3)
ax2.plot([i for i in range(len(time_per_iter_))], time_per_iter_)
ax2.set_xlabel("iteration number")
ax2.set_ylabel("Duration in seg per iteration")

ax3 = fig.add_subplot(2,3,4, projection='3d')
ax3.scatter([blob.herbo for blob in blobs], [blob.carno for blob in blobs],[blob.speed for blob in blobs], s=5, alpha=0.5, c='k')
ax3.set_xlim(0,1)
ax3.set_ylim(0,1)
ax3.set_zlim(0,1)
ax3.set_xlabel("herbo")
ax3.set_ylabel("carno")
ax3.set_zlabel("speed")

ax4 = fig.add_subplot(2,3,5, projection='3d')
ax4.plot(popu_stat[200:], popu_stat[100:-100], popu_stat[:-200])

print(np.mean([blob.number_of_babies for blob in blobs]))

print(np.mean([blob.energy_for_babies for blob in blobs]))

plt.show()
