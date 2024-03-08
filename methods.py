import numpy as np
import random

def noise(sigma:float=0.015)->float:
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
    offens : float                                                               # EEECCCXXX
    energy_for_babies: float # 0-1 parameter that indicates the number of babies
    number_of_babies: float # 0-1 parameter that establishes the probability of moving randomly in case there are no reasons for doing it
    curiosity: float  # 0-1 parameter that establishes the probability of moving randomly in case there are no reasons for doing it
    agressive: float  # 0-1 __-__-__-__-__
    collab: float #0-1 parameter that establishes how likely will share its energy with other blobs with similar stats
    skin: tuple[int,int,int]    # (0-255,0-255,0-255) color that blob will have and babies will inherit

    def __init__(self, x=None, y=None, energy=None, carno=None, herbo=None, speed=None, age=None, vision=None, offens=None, 
                 energy_for_babies=None, number_of_babies=None, curiosity=None, agressive=None, colab=None, skin=None)->None:
            
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
            self.skin = skin 

    def get_skin(self)->tuple[int,int,int]:
        """Return the skin color by default or by lineage"""
        if self.skin is None : return (compress(255*self.carno,255), compress(255*self.herbo,255), 0)
        return self.skin
    
    def show_features(self)->None: 
        """Print the parameters of Self"""
        for key, value in self.__dict__.items():
            print(f"    {key}: {value}")

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
                self.x = (self.x + dx) % grid.dim
                self.y = (self.y + dy) % grid.dim
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
                                         +0.2*len(grid.get_neighbours_dist(self.x, self.y, 2)) + 0.1*len(grid.get_neighbours_dist(self.x, self.y, 3))):
            self.energy += 0.7
        self.age +=1
        
        self.energy -= metabo*(1+ (self.herbo+self.carno+self.speed+self.vision+self.offens)/5 ) 

    def is_alive(self)->None:
        """Checks if blob remains alive or is already dead"""
        return self.energy > 0 and self.age < 300
    
    def reproduce(self, giving_birth_cost=1.2)->list['Blob']:
        """If blob pass a energy threshold, a new Blob is born with some mutations"""
        babies_energy = self.energy * self.energy_for_babies
        self.energy -= babies_energy*giving_birth_cost
        babies = []
        babies_num = int(self.number_of_babies//0.15)
        for _ in range(babies_num):
            babies.append(Blob(self.x+random.randint(-1, 1), self.y+random.randint(-1, 1), energy= babies_energy/babies_num, carno=compress(self.carno+noise()), 
                    herbo=compress(self.herbo+noise()), speed=compress(self.speed+noise()), age=1, 
                    vision=compress(self.vision + noise()), offens=compress(self.offens + noise()), 
                    energy_for_babies=compress(self.energy_for_babies+noise()), number_of_babies=compress(self.number_of_babies+noise()), 
                    curiosity=compress(self.curiosity+noise()), agressive=compress(self.agressive+noise()), colab=compress(self.colab+noise()),
                    skin=self.skin))

        return babies
