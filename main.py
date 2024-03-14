import numpy as np
import random

def noise(sigma:float=0.01)->float:
    """Encapsulates the generation of a random number with triangular distribution"""
    if sigma < 0 : sigma=0
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

def array_dist(array1, array2, p=2):
    """Calculates the p-norm between input vectors"""
    if len(array1) != len(array2):
        raise ValueError("Arrays must have the same length")
    powered_diff_sum = sum(abs(x - y) ** p for x, y in zip(array1, array2))
    return powered_diff_sum ** (1/p)

def witch(x:float)->float:
    """Compute the Witch of Agnesi of the given input, i.e. f(x)=1/(1+x^2)"""
    return 1/(1+x*x)

class Grid:
    """Consists in the grid where blobs move and interact"""
    dim : int   # lenght of grids side -> Grid has dim*dim shape
    dic : dict  # dictionary which stores the blobs, keys are tuple (x,y) of non empty tiles
                #   and values list with those blobs which are at that position

    def __init__(self, blob_list:list['Blob'], dim:int)->None:
        self.dic = {}
        self.dim = dim
        for blob in blob_list:
            if (blob.x%self.dim, blob.y%self.dim) in self.dic:
                self.dic[(blob.x%self.dim, blob.y%self.dim)].append(blob)
            else: 
                self.dic[(blob.x%self.dim, blob.y%self.dim)] = [blob]

    def blobs_at_tile(self, c_x:int, c_y:int)->list['Grid']:
        """Returns list with blobs at given tile"""
        if (c_x%self.dim, c_y%self.dim) not in self.dic: return []
        else: return self.dic[(c_x%self.dim, c_y%self.dim)]

    def get_neighbours_dist(self, c_x:int, c_y:int, vrad:int)->list['Blob']:
        """Returns the neighbours of tile (c_x,c_y) at certain distance of vrad"""
        neighbours = []
        if vrad == 0: return self.dic[(c_x%self.dim, c_y%self.dim)]

        for j in range(c_y - vrad, c_y + 1 + vrad):
            neighbours.extend(self.blobs_at_tile( (c_x - vrad)%self.dim, j%self.dim ))
            neighbours.extend(self.blobs_at_tile( (c_x + vrad)%self.dim, j%self.dim ))

        for i in range(c_x - vrad + 1, c_x + vrad):
            neighbours.extend(self.blobs_at_tile( (i)%self.dim, (c_y + vrad)%self.dim ))
            neighbours.extend(self.blobs_at_tile( (i)%self.dim, (c_y - vrad)%self.dim ))

        return neighbours

    def update(self, blob_list:list['Blob'])->None:
        """Update sel.dic because the blobs have moved"""
        self.dic = {}
        for blob in blob_list:
            if (blob.x%self.dim, blob.y%self.dim) in self.dic:
                self.dic[(blob.x%self.dim, blob.y%self.dim)].append(blob)
            else: 
                self.dic[(blob.x%self.dim, blob.y%self.dim)] = [blob]


class Blob:
    x : int         # horizontal position 
    y : int         # vertical position
    energy : int    # 0-100 amount of energy stored 
    
    carno : float   # 0-1 how many energy gets from feeding from preys
    herbo : float   # 0-1 how many energy gets from the surroundings
    
    speed : float   # 0-1 in averadge how many tiles moves per iteration
    vision : float  # 0-1 divided by 0.15 is the how far it sees
    
    age: int        # ** the current age of the blob 
    
    offens : float  # 0-1 how likely could eat a blob
    defens : float  # 0-1 how likely could survive a attack of a predator
    
    energy_for_babies: float # 0-1 indicates how much energy each baby will have
    number_of_babies: float # 0-1 indicates the number of babies that they will have
    
    curios: float  # 0-1 parameter that establishes the probability of moving randomly in case there are no reasons for doing it
    agress: float  # 0-1 weight that regulates how likely is to stalk another blob
    colab: float #0-1 parameter that establishes how likely will share its energy with other blobs with similar stats
    skin: tuple[int,int,int]    # (0-255,0-255,0-255) color that blob will have and babies will inherit
    
    fav_meal: list[float] # Array with independent values used as parameters for blob comparison

    def __init__(self, x=None, y=None, energy=None, carno=None, herbo=None, speed=None, age=None, vision=None, offens=None, defens=None,
                 energy_for_babies=None, number_of_babies=None, curiosity=None, agress=None, colab=None, skin=None, fav_meal=None)->None:
            
            # Each variable has a predeterminate value unless one is specified 
            self.x = 0 if x is None else x
            self.y = 0 if y is None else y
            self.energy = random.randint(20,80) if energy is None else energy
            self.age = random.randint(1,500) if age is None else age

            self.carno= random.uniform(0.1, 0.9) if carno is None else carno
            self.herbo = random.uniform(0.1, 0.9) if herbo is None else herbo
            
            self.speed = random.uniform(0.1, 0.9) if speed is None else speed
            self.vision = random.uniform(0.1, 0.9) if vision is None else vision
            
            self.offens = random.uniform(0.1, 0.9) if offens is None else offens
            self.defens = random.uniform(0.1, 0.9) if defens is None else defens
            
            self.energy_for_babies = random.uniform(0.1, 0.9) if energy_for_babies is None else energy_for_babies
            self.number_of_babies = random.uniform(0.1, 0.9) if number_of_babies is None else number_of_babies
            
            self.curiosity = random.uniform(0, 1) if curiosity is None else curiosity
            self.agress = random.uniform(0, 1) if agress is None else agress
            self.colab = random.uniform(0, 1) if colab is None else colab
            self.skin = skin 
            
            self.fav_meal = [random.uniform(0, 1) for _ in range(6)] if fav_meal is None else fav_meal


    def get_skin(self)->tuple[int,int,int]:
        """Return the skin color by default or by lineage"""
        if self.skin is None : return (compress(255*self.carno,255), compress(255*self.herbo,255), 0)
        return self.skin
    
    def show_features(self)->None: 
        """Print the parameters of Self"""
        for key, value in self.__dict__.items():
            print(f"    {key}: {value}")
    
    def anatomy(self)->list[float]:
        """Return list with physical features of Self"""
        return [self.carno, self.herbo, self.speed, self.vision, self.offens, self.defens]


    def compute_next_move(self, grid:'Grid')->tuple[int,int]: 
        """Compute a factor which determines how Self will move"""
        dx_prop , dy_prop = 0,0
        for k in range(1, int(1 + self.vision//0.3)):
            for blobi in grid.get_neighbours_dist(self.x, self.y, k):

                if abs( blobi.agress*witch(array_dist(self.anatomy(), blobi.fav_meal)) - self.agress*witch(array_dist(blobi.anatomy(), self.fav_meal)) ) > 0.1: #peligro
                    if blobi.agress*witch(array_dist(self.anatomy(), blobi.fav_meal)) > self.agress*witch(array_dist(blobi.anatomy(), self.fav_meal)) : 
                        dx_prop += 4*mod_dist(self.x, blobi.x, grid.dim)/k
                        dy_prop += 4*mod_dist(self.y, blobi.y, grid.dim)/k
                    else: 
                        dx_prop -= 4*mod_dist(self.x, blobi.x, grid.dim)/k
                        dy_prop -= 4*mod_dist(self.y, blobi.y, grid.dim)/k
                else: # amigos
                    dx_prop -= blobi.herbo*mod_dist(self.x,blobi.x,grid.dim)/k 
                    dy_prop -= blobi.herbo*mod_dist(self.y,blobi.y,grid.dim)/k

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
        if random.random() < self.speed: 

            dx , dy = self.compute_next_move(grid)

            # Update position with periodic boundary conditions
            if dx != 0 or dy != 0:
                self.x = (self.x + dx) % grid.dim
                self.y = (self.y + dy) % grid.dim
                self.energy -= movement_energy

    def fight(self, blob:'Blob')->None:
        """Self and a Blob fight and the result is updated"""
        if self.energy>0 and blob.energy>0 and self!=blob and (random.random() < self.agress or random.random() < blob.agress): # check if both are alive

                # What if they do not want to fight??
                # We should implement another function based in how likely is that they decide to fight. 
                # Therefore, the violent interaction would take place only when at least one of them wants to.
                # The following is just a basic implementation using the agress parameter
            if (blob.offens + self.offens)*random.random() < blob.offens :  # blob eats self
                if (blob.offens + self.defens)*random.random() < blob.offens:
                        blob.energy += self.energy * blob.carno * witch(array_dist(self.anatomy(), blob.fav_meal))
                        self.energy = -100
            
            else:       # self eats blob 
                if (self.offens + blob.defens)*random.random() < self.offens:
                        self.energy += blob.energy * self.carno * witch(array_dist(blob.anatomy(), self.fav_meal)) 
                        blob.energy = -100

    def vital(self, grid:'Grid', metabo:float = 0.1, herboGain = 1)->None:
        """
            Blob gains energy from surrondings with probability based on herbo level
            If multiple blobs are in the same tile the energy is distributed among them
        """
        if random.random() < self.herbo/(len(grid.blobs_at_tile(self.x, self.y))+0.2*len(grid.get_neighbours_dist(self.x, self.y, 1))
                                         +0.05*len(grid.get_neighbours_dist(self.x, self.y, 2)) ):
            self.energy += herboGain
        self.age +=1

        for blobi in grid.get_neighbours_dist(self.x, self.y, 1):
            if blobi.energy > 0 and array_dist(blobi.anatomy(), self.anatomy()) < 0.01 and blobi.energy < self.energy:
                # si es necesario implementar que se pierde energia en el intercambio
                transfer = self.colab * (self.energy-blobi.energy)
                self.energy -= transfer
                blobi.energy += transfer
        
        self.energy -= metabo*( sum(self.anatomy())/len(self.anatomy()) )

    def is_alive(self, maxAge=300)->None:
        """Checks if blob remains alive or is already dead"""
        return self.energy > 0 and self.age < maxAge
    
    def reproduce(self, giving_birth_cost=1.2, geneticVar = 0.01)->list['Blob']:
        """If blob pass a energy threshold, a new Blob is born with some mutations"""
        babies_energy = self.energy * self.energy_for_babies
        self.energy -= babies_energy*giving_birth_cost
        babies = []
        babies_num = int(self.number_of_babies//0.15)
        for _ in range(babies_num):
            babies.append(Blob(self.x+random.randint(-1, 1), self.y+random.randint(-1, 1), energy= babies_energy/babies_num, carno=compress(self.carno+noise(geneticVar)), 
                    herbo=compress(self.herbo+noise(geneticVar)), speed=compress(self.speed+noise(geneticVar)), age=1, 
                    vision=compress(self.vision + noise(geneticVar)), offens=compress(self.offens + noise(geneticVar)), defens = compress(self.defens + noise(geneticVar)), 
                    energy_for_babies=compress(self.energy_for_babies+noise(geneticVar)), number_of_babies=compress(self.number_of_babies+noise(geneticVar)), 
                    curiosity=compress(self.curiosity+noise(geneticVar)), agress=compress(self.agress+noise(geneticVar)), colab=compress(self.colab+noise(geneticVar)),
                    skin=self.skin, fav_meal= [compress(stat + noise(geneticVar)) for stat in self.fav_meal]))

        return babies




import pygame
import random
import matplotlib.pyplot as plt
import numpy as np
import time 
from sklearn.decomposition import PCA
# from ECmethods import Blob, Grid, compress

# Parameters of simulation
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 800
CELL_SIZE = 8
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
herbogain = 1 # <<<<<
maxage = 150 # <<<<<
gen_var = 0.05 # <<<<<

# Statistics 
popu_stat = []
speed_stat = []
herbo_stat = []
carno_stat = []
vision_stat = []
offens_stat = []
time_per_iter_ = []
deaths = [0, 0, 0]

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
        if event.type == pygame.QUIT or blobs==[] :
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
        time_move = time.time()
        for blob in blobs:
            blob.move(grid)
        # print("move: ", time.time() - time_move, end=' ')
        
        # Update the grid with the new positions
        # time_update = time.time()
        grid.update(blobs)
        # print("update: ", time.time() - time_update, end=' ')

        # Let blobs gain energy form the enviroment 
        time_vital = time.time()
        for blob in blobs:
            blob.vital(grid, metabolism, herbogain)
        # print("vital: ", time.time() - time_vital, end=' ')

        # time_carno = time.time()
        # Let blobs depredate each other
        for blob in blobs:
            if blob.energy > 0:
                neighbours = grid.blobs_at_tile(blob.x, blob.y)
                for neig in neighbours:
                    blob.fight(neig)
        # print("carno: ", time.time() - time_carno, end=' ')

        # time_repro = time.time()
        # Let the blobs feed and check if they may reproduce
        for blob in blobs:
            if blob.energy > 0 and blob.energy >= energy_to_reproduce:
                babies = blob.reproduce(geneticVar=gen_var)
                blobs.extend(babies)
                count_num_baby += len(babies)
        # print("rpro: ", time.time() - time_repro, end=' ')

        # time_remove = time.time()
        # Remove dead blobs
        blobs = [blob for blob in blobs if blob.is_alive(maxage)]
        # print("remove: ", time.time() - time_remove, end=' ')

                # Refresh the grid to the last update
        time_update = time.time()
        grid.update(blobs)
        # print("update: ", time.time() - time_update, end=' ')
            
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
ax3.scatter([blob.offens for blob in blobs], [blob.defens for blob in blobs], [blob.carno for blob in blobs], c=[blob.herbo for blob in blobs], s=5, alpha=0.5)
# ax3.scatter([blob.fav_meal[4] for blob in blobs], [blob.fav_meal[5] for blob in blobs], [blob.fav_meal[2] for blob in blobs], c='red', s=5, alpha=0.5)
ax3.set_xlim(0,1)
ax3.set_ylim(0,1)
ax3.set_zlim(0,1)
ax3.set_xlabel("offens")
ax3.set_ylabel("defens")
ax3.set_zlabel("carno")
# ax3.quiver([blob.carno for blob in blobs], [blob.vision for blob in blobs], [blob.speed for blob in blobs], [blob.fav_meal[0] for blob in blobs], [blob.fav_meal[3] for blob in blobs], [blob.fav_meal[2] for blob in blobs], length=0.1, normalize=True)

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
