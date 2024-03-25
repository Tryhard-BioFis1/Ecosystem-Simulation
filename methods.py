import numpy as np
import random

def noise(sigma:float=0.01)->float:
    """Encapsulates the generation of a random number with triangular distribution"""
    return np.random.normal(scale=sigma)

def compress(a:float, max_lim:float=1.0)->float:
    """
    Given a float 'a', if overpass the limits 0 (lower limit) or 'max_lim' 
    return the overpassed limit, else return 'a'
    """
    if a>max_lim: return max_lim
    if a<0: return 0.0
    return a

def mod_dist(a:int, b:int, n:int)->int:
    """Calculates the minimun SIGNED steps to get from a to b modulo n"""
    a_ , b_ = a%n , b%n
    dst = min(abs(b_-a_), n-abs(b_-a_))
    if (a_+dst)%n == b_ : return dst
    return -dst

def array_dist(array1:list[float], array2:list[float], p:int=2):
    """Calculates the p-norm between input vectors"""
    if len(array1) != len(array2):
        raise ValueError("Arrays must have the same length")
    powered_diff_sum = sum(abs(x - y) ** p for x, y in zip(array1, array2))
    return powered_diff_sum ** (1/p)

def witch(x:float)->float:
    """Compute the Witch of Agnesi of the given input, i.e. f(x)=1/(1+x^2)"""
    return 1/(1+x*x)

class Soil:
    def __init__(self, dim: int) -> None:
        self.dim = dim
        self.soil = np.full((dim, dim, 2), 0.5)  # Initialize soil with new and old values

    def get_value(self, x: int, y: int) -> tuple:
        """Returns the soil tuple (new value, old value) at given coordinates (x, y)"""
        x = x % self.dim
        y = y % self.dim
        return self.soil[x, y]

    def set_value(self, x: int, y: int, new_value: float) -> None:
        """Sets the new value of soil at given coordinates (x, y)"""
        x = x % self.dim
        y = y % self.dim
        self.soil[x, y, 0] = new_value  # Set the new value in the first layer

    def update(self) -> None:
        """Update the old value of soil with the new value"""
        self.soil[:, :, 1] = self.soil[:, :, 0]

    def grow(self, growth_rate: float = 0.01) -> None:
        """Update the new value of soil based on the sum of its and its neighbors' old values"""
        for i in range(self.dim):
            for j in range(self.dim):
                sum_neighbors = (
                    self.get_value(i, j)[1] +
                    self.get_value(i - 1, j)[1] +
                    self.get_value(i + 1, j)[1] +
                    self.get_value(i, j - 1)[1] +
                    self.get_value(i, j + 1)[1]
                )
                new_value = min(1, max(0, self.get_value(i, j)[0] + growth_rate * sum_neighbors))
                self.set_value(i, j, new_value)


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
        """Returns a list with blobs at given tile"""
        if (c_x%self.dim, c_y%self.dim) not in self.dic: return []
        else: return self.dic[(c_x%self.dim, c_y%self.dim)]

    def get_neighbours_dist(self, c_x:int, c_y:int, vrad:int)->list['Blob']:
        """Returns a list with blobs at the neighbouring tiles of (c_x,c_y) at certain distance of vrad"""
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
        """Update sel.dic because the blobs may have moved"""
        self.dic = {}
        for blob in blob_list:
            if (blob.x%self.dim, blob.y%self.dim) in self.dic:
                self.dic[(blob.x%self.dim, blob.y%self.dim)].append(blob)
            else: 
                self.dic[(blob.x%self.dim, blob.y%self.dim)] = [blob]


class Blob:
    x : int         # horizontal position 
    y : int         # vertical position
    energy : float    # 0-100 amount of energy stored 
    
    phago : float   # 0-1 how many energy gets from feeding from preys
    phyto : float   # 0-1 how many energy gets from the surroundings
    
    speed : float   # 0-1 in average how many tiles moves per iteration
    vision : float  # 0-1 divided by 0.15 is the how far it sees
    
    age: float        # ** the current age of the blob 
    
    offens : float  # 0-1 how likely could eat a blob
    defens : float  # 0-1 how likely could survive a attack of a predator
    
    energy_for_babies: float # 0-1 indicates how much energy each baby will have
    number_of_babies: float # 0-1 indicates the number of babies that they will have
    
    curios: float  # 0-1 parameter that establishes the probability of moving randomly in case there are no reasons for doing it
    agress: float  # 0-1 weight that regulates how likely is to stalk another blob
    colab: float #0-1 parameter that establishes how likely will share its energy with other blobs with similar stats
    skin: tuple[int,int,int]    # (0-255,0-255,0-255) color that blob will have and babies will inherit
    
    fav_meal: list[float] # Array with independent values used as parameters for blob comparison

    def __init__(self, x=None, y=None, energy=None, phago=None, phyto=None, speed=None, age=None, vision=None, offens=None, defens=None,
                 energy_for_babies=None, number_of_babies=None, curiosity=None, agress=None, colab=None, skin=None, fav_meal=None)->None:
            
            # Each variable has a predeterminate value unless one is specified 
            self.x = 0 if x is None else x
            self.y = 0 if y is None else y
            self.energy = random.uniform(20,80) if energy is None else energy
            self.age = random.uniform(1,500) if age is None else age

            self.phago= random.uniform(0.1, 0.9) if phago is None else phago
            self.phyto = random.uniform(0.1, 0.9) if phyto is None else phyto
            
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
        if self.skin is None : return (compress(255*self.phago,255), compress(255*self.phyto,255), 0)
        return self.skin
    
    def show_features(self)->None: 
        """Print the parameters of Self"""
        for key, value in self.__dict__.items():
            print(f"    {key}: {value}")
    
    def anatomy(self)->list[float]:
        """Return list with physical features of Self"""
        return [self.phago, self.phyto, self.speed, self.vision, self.offens, self.defens]

    def compute_next_move(self, grid:'Grid')->tuple[int,int]: 
        """Compute a factor which determines how Self will move"""
        dx_prop , dy_prop = 0.0, 0.0
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
                    dx_prop -= blobi.phyto*mod_dist(self.x,blobi.x,grid.dim)/k 
                    dy_prop -= blobi.phyto*mod_dist(self.y,blobi.y,grid.dim)/k

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

    def move(self, grid:'Grid', movement_energy:float=0.5)->None:
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
                        blob.energy += self.energy * blob.phago * witch(array_dist(self.anatomy(), blob.fav_meal))
                        self.energy = -100
            
            else:       # self eats blob 
                if (self.offens + blob.defens)*random.random() < self.offens:
                        self.energy += blob.energy * self.phago * witch(array_dist(blob.anatomy(), self.fav_meal)) 
                        blob.energy = -100

    def vital(self, soil: 'soil', grid:'Grid', metabo:float = 0.1, phytoGain:float = 1.0)->None:
        """
            Blob gains energy from surrondings with probability based on phyto level
            If multiple blobs are in the same tile the energy is distributed among them
        """
        if random.random() < self.phyto/(len(grid.blobs_at_tile(self.x, self.y))+0.2*len(grid.get_neighbours_dist(self.x, self.y, 1))
                                         +0.05*len(grid.get_neighbours_dist(self.x, self.y, 2)) ):
            self.energy += phytoGain*soil.get_value(self.x, self.y)[1]
            soil.set_value(self.x, self.y, max(0, soil.get_value(self.x, self.y)[0] - 0.3*self.phyto))

        self.age +=1

        for blobi in grid.get_neighbours_dist(self.x, self.y, 1):
            if blobi.energy > 0 and array_dist(blobi.anatomy(), self.anatomy()) < 0.01 and blobi.energy < self.energy:
                # si es necesario implementar que se pierde energia en el intercambio
                transfer = self.colab * (self.energy-blobi.energy)
                self.energy -= transfer
                blobi.energy += transfer
        
        self.energy -= metabo*( sum(self.anatomy())/len(self.anatomy()) )
    
    def is_alive(self, death_cause_list:list[int], maxAge=300)->None: 
        """Checks if blob remains alive or is already dead"""
        if self.energy < -90: death_cause_list[0] += 1  #died from depredation
        elif self.energy <= 0: death_cause_list[1] += 1  #died from starvation
        elif self.age >= maxAge: death_cause_list[2] += 1 #died because of age
        return self.energy > 0 and self.age < maxAge  
    
    def reproduce(self, giving_birth_cost:float=1.2, geneticVar:float=0.01)->list['Blob']:
        """If blob pass a energy threshold, a new Blob is born with some mutations"""
        babies_energy = self.energy * self.energy_for_babies
        self.energy -= babies_energy*giving_birth_cost
        babies = []
        babies_num = int(self.number_of_babies//0.15)
        for _ in range(babies_num):
            babies.append(Blob(self.x+random.randint(-1, 1), self.y+random.randint(-1, 1), energy= babies_energy/babies_num, phago=compress(self.phago+noise(geneticVar)), 
                    phyto=compress(self.phyto+noise(geneticVar)), speed=compress(self.speed+noise(geneticVar)), age=1, 
                    vision=compress(self.vision + noise(geneticVar)), offens=compress(self.offens + noise(geneticVar)), defens = compress(self.defens + noise(geneticVar)), 
                    energy_for_babies=compress(self.energy_for_babies+noise(geneticVar)), number_of_babies=compress(self.number_of_babies+noise(geneticVar)), 
                    curiosity=compress(self.curiosity+noise(geneticVar)), agress=compress(self.agress+noise(geneticVar)), colab=compress(self.colab+noise(geneticVar)),
                    skin=self.skin, fav_meal= [compress(stat + noise(geneticVar)) for stat in self.fav_meal]))

        return babies
