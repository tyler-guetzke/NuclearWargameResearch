import random
from tkinter.ttk import Style
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits import mplot3d
import math
from sklearn.utils import shuffle
from fitter import Fitter
#import scipy.integrate as integrate
#from scipy.interpolate import interp1d
from pyproj import Transformer
from scipy import stats
import scipy.stats
from scipy.stats import nbinom, poisson, geom
from scipy.stats import gaussian_kde

from tabulate import tabulate
import tkinter

import time

#Set up nuke class (parameters concerning the nuke contents)
class Nuke:
    def __init__(self, delivery, nuke_center, Yield):
        nsnw_dict = {'Bomb':(100,Yield), 'ICBM':(100,Yield), 'Submarine':(100,Yield)}
        index = np.array(['CEP','Yield'])

        nsnw_df = pd.DataFrame(nsnw_dict)
        self.nsnw_df = nsnw_df.set_index(index)

        self.cep, self.Yield = nsnw_dict[delivery]
        sigma_sq = self.cep * 1/math.sqrt(math.log(4))

        x_coord, y_coord, z_coord = nuke_center

        self.x_mean = x_coord
        self.x_var = sigma_sq

        self.y_mean = y_coord
        self.y_var = sigma_sq

        self.HOB_mean = z_coord
        self.HOB_var = 16

        self.V = np.array([
            [self.x_var, 0, 0],
            [0, self.y_var, 0],
            [0, 0, self.HOB_var]])
        self.A = np.array([self.x_mean, self.y_mean, self.HOB_mean])

    def drop_nuke(self, n_drops = 1):
        self.nsnw_drop = np.random.multivariate_normal(self.A, self.V, n_drops)
        self.displacement = math.sqrt((self.nsnw_drop[0,0] - self.x_mean)**2 + (self.nsnw_drop[0,1] - self.y_mean)**2 +(self.nsnw_drop[0,2] - self.HOB_mean)**2)

    def plot_nuke(self):
        fig = plt.figure()
        plt.clf()

        ax = fig.add_subplot(111, projection='3d')
        ax.scatter3D(self.nsnw_drop[:,0], self.nsnw_drop[:,1], self.nsnw_drop[:,2])
        ax.scatter3D(self.x_mean, self.y_mean, self.HOB_mean) 
        ax.set_xlabel('X Coordinates (Meters)')
        ax.set_ylabel('Y Coordinates (Meters)')
        ax.set_zlabel('Z Coordinates (Meters)')
        ax.set_title('Actual Ground Zero (AGZ)')
        ax.set_xlim(self.x_mean - 50, self.x_mean + 50)
        ax.set_ylim(self.y_mean - 50, self.y_mean + 50)
        ax.set_zlim(self.HOB_mean - 20, self.HOB_mean + 20)

        if len(self.nsnw_drop) == 1:
            x = self.nsnw_drop[0, 0] + 1
            y = self.nsnw_drop[0, 1] + 1
            z = self.nsnw_drop[0, 2] + 1
            coord = '(' + str(x.round(2)) + ', ' + str(y.round(2)) + ', ' + str(z.round(2)) + ')'
            ax.text(x, y, z, coord)
            
            DGZ_coord = '(' + str(self.x_mean) + ', ' + str(self.y_mean) + ', ' + str(self.HOB_mean) + ')'
            ax.text(self.x_mean + 1, self.y_mean + 1, self.HOB_mean + 1, DGZ_coord)

        plt.show()

    def nuke_effects(self, d):
        scaled_yield = self.Yield ** (1/3)
        scaled_d_ft = np.array([(d[i]*3.28084)/scaled_yield for i in range (len(d))])
        scaled_HOB_ft = np.array([self.nsnw_drop[i,2]*3.28084/scaled_yield for i in range (len(self.nsnw_drop))])
        r = scaled_d_ft/1000

        gr = np.array([math.sqrt(abs((d[i]**2) - self.nsnw_drop[:,2]**2))for i in range (len(d))])
        z = self.nsnw_drop[:, 2]/gr

        def blast_effects(scaled_yield, z, scaled_HOB, r):
            #Taken directly from past project but changed a few variable names for easier interpritation
            a_z = 1.22 - (3.908*(z**2))/(1+(810.2*(z**5)))
            b_z = 2.321 + ((6.195*(z**18))/(1+(1.641*(z**18)))) - ((.03831*(z**17))/(1+(.02415*(z**17)))) + ((.6692)/(1+(4164*(z**8))))
            c_z = 4.153 - ((1.149*z**18)/(1+(1.641*z**18))) - ((1.1)/(1+(2.771*z**2.5)))
            d_z = -4.166 + 25.76*z**1.75/(1+1.382*z**18) + 8.275*z/(1+3.219*z)
            e_z = 1 - 0.004642*z**18/(1+0.003886*z**18)
            f_z = 0.6096 + 2.879*z**9.25/(1+2.359*z**14.5) - 17.15*z**2/(1+71.66*z**3)
            g_z = 1.83 +5.361*z**2/(1+0.3139*z**6)
            h_z = -(64.67*z**5 + 0.2905)/(1+441.5*z**5) - 1.389*z/(1+49.03*z**5) + 8.808*z**1.5/(1+154.5*z**3.5) + 1.094*scaled_yield**2/((0.7812*10**9 - 1.234*10**5*scaled_yield + 1201*scaled_yield**1.5+scaled_yield**2)*(1+0.002*scaled_HOB))
            p_Y = 0.000629 - 2.121*scaled_HOB**2/(794300+scaled_HOB**4.3)
            q_Y = 5.18 + 8.864*scaled_HOB**3.5/(3.788 *10**6 + scaled_HOB**4)
            pressure = 10.47/r**a_z + b_z/r**c_z + (d_z * e_z)/(1+f_z*r**g_z) + h_z + p_Y/r**q_Y
            return pressure

        scaled_d_cm = np.array([(d[i]*100) for i in range (len(d))])
        
        def thermal_effects(scaled_d_cm):
            f = .35
            thermal = ((f * self.Yield)/(4 * math.pi * (scaled_d_cm ** 2)) * (10 ** 12))
            return thermal

        def radiation_effects():
            pass

        blast = blast_effects(scaled_yield, z, scaled_HOB_ft, r)
        thermal = thermal_effects(scaled_d_cm)
        
        return blast, thermal

    def creating_df(self):
        nuke_df = pd.DataFrame()
        nuke_df['X'] = self.nsnw_drop[:,0]
        nuke_df['Y'] = self.nsnw_drop[:,1]
        nuke_df['Z'] = self.nsnw_drop[:,2]
        return nuke_df

    def plotting_df(self, nuke_df, latitude, longitude):
        #https://gis.stackexchange.com/questions/78838/converting-projected-coordinates-to-lat-lon-using-python
        def xy_to_latlong(x,y):
            transformer = Transformer.from_crs("epsg:3857", "epsg:4326")
            lat,lon = transformer.transform(x,y)
            return (lat,lon)

        nuke_df['coordinates'] = list(zip(nuke_df['X'], nuke_df['Y']))

        latlon = [xy_to_latlong(x, y) for x, y in nuke_df['coordinates'] ]

        nuke_df['latlon'] = latlon
        # Split that column out into two separate columns - latitude and longitude
        nuke_df[['latitude', 'longitude']] = nuke_df['latlon'].apply(pd.Series)

        nuke_df['latitude'] = nuke_df['latitude'].apply(lambda x: x + latitude)
        nuke_df['longitude'] = nuke_df['longitude'].apply(lambda x: x + longitude)

        #https://towardsdatascience.com/creating-an-interactive-map-in-python-using-bokeh-and-pandas-f84414536a06
        def latlong_to_mercator(lat, lon):
            r_major = 6378137.000
            x = r_major * np.radians(lon)
            scale = x/lon
            y = 180.0/np.pi * np.log(np.tan(np.pi/4.0 + 
                lat * (np.pi/180.0)/2.0)) * scale
            return (x, y)
        # Define coord as tuple (lat,long)
        nuke_df['coordinates'] = list(zip(nuke_df['latitude'], nuke_df['longitude']))

        # Obtain list of mercator coordinates
        mercators = [latlong_to_mercator(x, y) for x, y in nuke_df['coordinates'] ]

        nuke_df['mercator'] = mercators
        # Split that column out into two separate columns - mercator_x and mercator_y
        nuke_df[['mercator_x', 'mercator_y']] = nuke_df['mercator'].apply(pd.Series)

        nuke_df = nuke_df.drop(labels = ['mercator', 'latlon', 'coordinates'], axis = 1)
        return nuke_df

#Set up unit class (parameters concerning the units contents)
class Unit:
    def __init__(self, commander, unittype, phase, individual_unit):
        self.commander = commander
        self.phase = phase

        if unittype == 'IBCT':
            self.unit_dict = {'HQ CO': {'Soldier' : 80, 'Truck' : 1, 'Vehicle' : 19}, 
            'IN CO': {'Soldier' : 120, 'Truck' : 1, 'Vehicle' : 3},
            'Mounted CO': {'Soldier' : 120, 'Truck' : 1, 'Vehicle' : 8},
            'FA CO': {'Soldier' : 100, 'Truck' : 4, 'Vehicle' : 21},
            'EN CO': {'Soldier' : 100, 'Truck' : 10, 'Vehicle' : 10},
            'SPT CO': {'Soldier' : 100, 'Truck' : 0, 'Vehicle' : 8},
            'FWD SPT CO': {'Soldier' : 100, 'Truck' : 10, 'Vehicle' : 6}
            }
        elif unittype == 'ABCT':
            self.unit_dict = {'HQ CO': {'Soldier' : 120, 'Tank' : 0, 'Truck' : 1, 'Vehicle' : 19, 'LA' : 0}, 
            'CAV CO': {'Soldier' : 100, 'Tank' : 0, 'Truck' : 1, 'Vehicle' : 17, 'LA' : 12},
            'AR CO': {'Soldier' : 60, 'Tank' : 12, 'Truck' : 1, 'Vehicle' : 3, 'LA' : 0},
            'FA CO': {'Soldier' : 60, 'Tank' : 0, 'Truck' : 1, 'Vehicle' : 6, 'LA' : 6},
            'EN CO': {'Soldier' : 100, 'Tank' : 0, 'Truck' : 10, 'Vehicle' : 10, 'LA' : 2},
            'SPT CO': {'Soldier' : 100, 'Tank' : 0, 'Truck' : 0, 'Vehicle' : 8, 'LA' : 0},
            'FWD SPT CO': {'Soldier' : 100, 'Tank' : 0, 'Truck' : 10, 'Vehicle' : 6, 'LA' : 0}
            }
        elif unittype == 'SBCT':
            self.unit_dict = {'HQ CO': {'Soldier' : 120, 'Truck' : 1, 'Vehicle' : 19, 'LA' : 0}, 
            'SBCT IN CO': {'Soldier' : 100, 'Truck' : 1, 'Vehicle' : 17, 'LA' : 12},
            'FA CO': {'Soldier' : 100, 'Truck' : 4, 'Vehicle' : 21, 'LA' : 0},
            'EN CO': {'Soldier' : 100, 'Truck' : 10, 'Vehicle' : 10, 'LA' : 0},
            'SPT CO': {'Soldier' : 100, 'Truck' : 0, 'Vehicle' : 8, 'LA' : 0},
            'FWD SPT CO': {'Soldier' : 100, 'Truck' : 10, 'Vehicle' : 6, 'LA' : 0}
            }
        elif unittype == 'CAB':
            self.unit_dict = {'HQ CO': {'Soldier' : 120, 'Truck' : 1, 'Vehicle' : 19, 'Helicopter' : 0}, 
            'HELO CO': {'Soldier' : 70, 'Truck' : 1, 'Vehicle' : 3, 'Helicopter' : 8},
            'SPT CO': {'Soldier' : 100, 'Truck' : 10, 'Vehicle' : 10, 'Helicopter' : 0}
            }
    
        if self.commander == 'IBCT':
            self.CO_list = np.array(['HQ CO', #HQ CO
            'HQ CO', 'IN CO', 'IN CO', 'IN CO', 'IN CO', #IN BN
            'HQ CO', 'IN CO', 'IN CO', 'IN CO', 'IN CO', #IN BN
            'HQ CO', 'IN CO', 'IN CO', 'IN CO', 'IN CO', #IN BN
            'HQ CO', 'Mounted CO', 'Mounted CO', 'IN CO', #CAV SQDN
            'HQ CO', 'FA CO', 'FA CO', 'FA CO', #FA BN
            'HQ CO', 'EN CO', 'EN CO', 'SPT CO', 'SPT CO', #BEB
            'FWD SPT CO', 'FWD SPT CO', 'FWD SPT CO', 'FWD SPT CO', 'FWD SPT CO', 'FWD SPT CO', 'SPT CO', 'SPT CO', 'SPT CO' #BSB
            ])
        elif self.commander == 'IN BN':
            self.CO_list = ['HQ CO', 'IN CO', 'IN CO', 'IN CO', 'IN CO'
            ]
        elif self.commander == 'CAV SQDN':
            self.CO_list = ['HQ CO', 'Mounted CO', 'Mounted CO', 'IN CO'
            ]
        elif self.commander == 'FA BN':
            self.CO_list = ['HQ CO', 'FA CO', 'FA CO', 'FA CO'
            ]
        elif self.commander == 'BEB':
            self.CO_list = ['HQ CO', 'EN CO', 'EN CO', 'SPT CO', 'SPT CO'
            ]
        elif self.commander == 'BSB':
            self.CO_list = ['FWD SPT CO', 'FWD SPT CO', 'FWD SPT CO', 'FWD SPT CO', 'FWD SPT CO', 'FWD SPT CO', 'SPT CO', 'SPT CO', 'SPT CO'
            ]
        elif self.commander == 'ABCT':
            self.CO_list = ['HQ CO', #HQ CO
            'HQ CO', 'AR CO', 'AR CO', 'CAV CO', #Combined Arms BN
            'HQ CO', 'AR CO', 'AR CO', 'CAV CO', #Combined Arms BN
            'HQ CO', 'AR CO', 'AR CO', 'CAV CO', #Combined Arms BN
            'HQ CO', 'FA CO', 'FA CO', 'FA CO', #FA BN
            'HQ CO', 'AR CO', 'CAV CO', 'CAV CO', 'CAV CO', #CAV BN
            'HQ CO', 'EN CO', 'EN CO', 'SPT CO', 'SPT CO', #BEB
            'FWD SPT CO', 'FWD SPT CO', 'FWD SPT CO', 'FWD SPT CO', 'FWD SPT CO', 'FWD SPT CO', 'SPT CO', 'SPT CO', 'SPT CO' #BSB
            ]
        elif self.commander == 'Combined Arms BN':
            self.CO_list = ['HQ CO', 'AR CO', 'AR CO', 'CAV CO'
            ]
        elif self.commander == 'CAV BN':
            self.CO_list = ['HQ CO', 'AR CO', 'CAV CO', 'CAV CO', 'CAV CO'
            ]
        elif self.commander == 'SBCT':
            self.CO_list = ['HQ CO', #HQ CO
            'HQ CO', 'SBCT IN CO', 'SBCT IN CO', 'SBCT IN CO', #SBCT IN BN
            'HQ CO', 'SBCT IN CO', 'SBCT IN CO', 'SBCT IN CO', #SBCT IN BN
            'HQ CO', 'SBCT IN CO', 'SBCT IN CO', 'SBCT IN CO', #SBCT IN BN
            'HQ CO', 'FA CO', 'FA CO', 'FA CO', #FA BN
            'HQ CO', 'EN CO', 'EN CO', 'SPT CO', 'SPT CO', #BEB
            'FWD SPT CO', 'FWD SPT CO', 'FWD SPT CO', 'FWD SPT CO', 'FWD SPT CO', 'FWD SPT CO', 'SPT CO', 'SPT CO', 'SPT CO' #BSB
            ]
        elif self.commander == 'SBCT IN BN':
            self.CO_list = ['HQ CO', 'AR CO', 'AR CO', 'CAV CO'
            ]
        elif self.commander == 'CAB':
            self.CO_list = ['HQ CO', 'HELO CO', 'HELO CO', 'HELO CO', #HELO BN
            'HQ CO', 'HELO CO', 'HELO CO', 'HELO CO', #HELO BN
            'HQ CO', 'HELO CO', 'HELO CO', 'HELO CO', #HELO BN
            'HQ CO', 'HELO CO', 'HELO CO', 'HELO CO', #HELO BN
            'HQ CO', 'SPT CO', 'SPT CO', 'SPT CO' #ASB
            ]
        elif self.commander == 'HELO BN':
            self.CO_list = ['HQ CO', 'HELO CO', 'HELO CO', 'HELO CO'
            ]
        elif self.commander == 'ASB':
            self.CO_list = ['HQ CO', 'SPT CO', 'SPT CO', 'SPT CO'
            ]
        elif self.commander == 'IN CO':
            self.CO_list = ['IN CO'
            ]
        elif self.commander == 'Mounted CO':
            self.CO_list = ['Mounted CO'
            ]
        elif self.commander == 'FA CO':
            self.CO_list = ['FA CO'
            ]
        elif self.commander == 'EN CO':
            self.CO_list = ['EN CO'
            ]
        elif self.commander == 'SPT CO':
            self.CO_list = ['SPT CO'
            ]
        elif self.commander == 'FWD SPT CO':
            self.CO_list = ['FWD SPT CO'
            ]
        elif self.commander == 'CAV CO':
            self.CO_list = ['CAV CO'
            ]
        elif self.commander == 'AR CO':
            self.CO_list = ['AR CO'
            ]
        elif self.commander == 'SBCT IN CO':
            self.CO_list = ['SBCT IN CO'
            ]
        elif self.commander == 'HELO CO':
            self.CO_list = ['HELO CO'
            ]
        
        if individual_unit == True:
            unique_co = []
            CO_list = []
            for co in self.CO_list:
                if co not in unique_co:
                    unique_co.append(co)
                    CO_list.append(f'{co} 1')
                else:
                    sum = unique_co.count(co)
                    CO_list.append(f'{co} {sum + 1}')
                    unique_co.append(co)
            
            self.CO_list = CO_list

        self.CO_list = shuffle(self.CO_list)

        unitcomp_df = pd.DataFrame(self.unit_dict)
        
        vuln_dict = {'Blast' : (10, 10, 10, 10, 10, 10), 'Thermal' : (10, 10, 10, 10, 10, 10), 'Radiation' : (10, 10, 10, 10, 10, 10)}
        vuln_df = pd.DataFrame(vuln_dict)
        index = np.array(['Soldier', 'Truck', 'Vehicle', 'Helicopter', 'LA', 'Tank'])
        self.vuln_df = vuln_df.set_index(index)

        self.blueforce_df = unitcomp_df.merge(self.vuln_df, left_index = True, right_index = True, how = 'left')

    def base_unit(self, co):
        'Strips numbers off the unit'
        striped_co = ''
        for char in co:
            if char.isalpha() == True or char == ' ':
                striped_co += char

        co = striped_co.rstrip(' ')
        return co

    def unit_size(self):
        unit_size_array = np.array(())
        for co in self.CO_list:
            
            co = self.base_unit(co)

            unit_size = 0
            for key,value in self.unit_dict[co].items():
                unit_size += value
            unit_size = np.array((unit_size))
            unit_size_array = np.append(unit_size_array, unit_size)
        return unit_size_array

    def unit_location(self, unit_center, radius, unit_size, i, th):
        #Functions to plot the unit
    
        def RandomPointInCircle(radius = 1, x = 0, y = 0, z = 0):
            r = radius * math.sqrt(random.random())
            theta = 2 * math.pi * random.random()
            return x + r * math.cos(theta), y + r * math.sin(theta), z

        def Replicate_N_Times(func, radius, x, y, z, Ntrials = 1000):
            dist = np.array([func(radius, x, y, z) for i in range(Ntrials)])
            return dist

        def rotate(center, old_x, old_y, angle):

            center_x, center_y, z = center

            new_x = center_x + math.cos(angle) * (old_x - center_x) - math.sin(angle) * (old_y - center_y)
            new_y = center_y + math.sin(angle) * (old_x - center_x) + math.cos(angle) * (old_y - center_y)
            return new_x, new_y

        x_center_coord, y_center_coord, z_center_coord = unit_center

        floor = (i // 4)

        if floor % 2 == 0:
            counter = 1
        else:
            counter = -1
        
        formation_loc = (i - (floor * 4))

        self.radius = radius

        if self.phase == 1:
            if formation_loc == 0:
                self.unit_center_x = x_center_coord + 1000
                self.unit_center_y = y_center_coord + (2000 * ((floor + 1) // 2) * counter)
            elif formation_loc == 1:
                self.unit_center_x = x_center_coord + 3000
                self.unit_center_y = y_center_coord + (2000 * ((floor + 1) // 2) * counter) 
            elif formation_loc == 2:
                self.unit_center_x = x_center_coord - 1000
                self.unit_center_y = y_center_coord + (2000 * ((floor + 1) // 2) * counter)
            elif formation_loc == 3:
                self.unit_center_x = x_center_coord - 3000
                self.unit_center_y = y_center_coord + (2000 * ((floor + 1) // 2) * counter)
        elif self.phase == 2:
            if formation_loc == 0:
                self.unit_center_x = x_center_coord
                self.unit_center_y = y_center_coord + (10000 * ((floor + 1) // 2) * counter)
            elif formation_loc == 1:
                self.unit_center_x = x_center_coord
                self.unit_center_y = y_center_coord - 2500 + (10000 * ((floor + 1) // 2) * counter) 
            elif formation_loc == 2:
                self.unit_center_x = x_center_coord
                self.unit_center_y = y_center_coord - 5000 + (10000 * ((floor + 1) // 2) * counter)
            elif formation_loc == 3:
                self.unit_center_x = x_center_coord
                self.unit_center_y = y_center_coord - 7500 + (10000 * ((floor + 1) // 2) * counter)
        elif self.phase == 3:
            if formation_loc == 0:
                self.unit_center_x = x_center_coord + 1000
                self.unit_center_y = y_center_coord + (2000 * ((floor + 1) // 2) * counter)
            elif formation_loc == 1:
                self.unit_center_x = x_center_coord + 3000
                self.unit_center_y = y_center_coord + (2000 * ((floor + 1) // 2) * counter) 
            elif formation_loc == 2:
                self.unit_center_x = x_center_coord - 1000
                self.unit_center_y = y_center_coord + (2000 * ((floor + 1) // 2) * counter)
            elif formation_loc == 3:
                self.unit_center_x = x_center_coord - 3000
                self.unit_center_y = y_center_coord + (2000 * ((floor + 1) // 2) * counter)
        elif self.phase == 4:
            if formation_loc == 0:
                self.unit_center_x = x_center_coord 
                self.unit_center_y = y_center_coord + ((self.radius * 4) * ((floor + 1) // 2) * counter)
            elif formation_loc == 1:
                self.unit_center_x = x_center_coord + self.radius * 2
                self.unit_center_y = y_center_coord - (self.radius * 2) + ((self.radius * 4) * ((floor + 1) // 2) * counter)
            elif formation_loc == 2:
                self.unit_center_x = x_center_coord - self.radius * 2
                self.unit_center_y = y_center_coord - (self.radius * 2) + ((self.radius * 4) * ((floor + 1) // 2) * counter)
            elif formation_loc == 3 and ((floor + 1) // 2) % 2 == 1:
                self.unit_center_x = x_center_coord + self.radius * 4
                self.unit_center_y = y_center_coord - (self.radius * 4) + ((self.radius * 4) * ((floor + 1) // 2) * counter)
            elif formation_loc == 3 and ((floor + 1) // 2) % 2 == 0:
                self.unit_center_x = x_center_coord - self.radius * 4
                self.unit_center_y = y_center_coord - (self.radius * 4) + ((self.radius * 4) * ((floor + 1) // 2) * counter)
        else:
            print('Needs a valid phase!')
        
        self.unit_center_z = z_center_coord

        #Generating the unit locations
        _unit_coordinates = Replicate_N_Times(RandomPointInCircle, radius, self.unit_center_x, self.unit_center_y, self.unit_center_z, unit_size)
        
        unit_coordinates = []
        for i in range(0, len(_unit_coordinates)):

            x = np.array([_unit_coordinates[i][0]])
            y = np.array([_unit_coordinates[i][1]])

            x_coord,y_coord = rotate(unit_center, x, y, theta)
            unit_coordinates.append([x_coord[0],y_coord[0],0])
        unit_coordinates = np.array(unit_coordinates)
        
        return unit_coordinates

    def plot_unit(self, units_df, nuke_df):
        #Plotting the circle
        
        #circle = np.array([(unit_center_x + radius * math.cos(i), unit_center_y + radius * math.sin(i), unit_center_z) for i in np.arange(0, math.pi * 2, 0.01)])

        fig = plt.figure()
        plt.clf()

        ax = fig.add_subplot(111, projection='3d')

        '''
        xline = np.linspace(nuke_df['X'], units_df['X'], 10)
        yline = np.linspace(nuke_df['Y'], units_df['Y'], 10)
        zline = np.linspace(nuke_df['Z'], units_df['Z'], 10)
        xt=np.transpose(xline)
        yt=np.transpose(yline)
        zt=np.transpose(zline)

        for i in range(len(xt)):
            ax.plot3D(xt[i], yt[i], zt[i],color='red')
        '''

        ax.scatter3D(nuke_df['X'], nuke_df['Y'], nuke_df['Z'])

        groups = units_df.groupby('Company')
        for name, group in groups:
            ax.scatter3D(group['X'], group['Y'], group['Z'], marker = 'o', label = name)

        ax.legend(title='Units', bbox_to_anchor=(1.05, 1), loc='upper left')

        ax.set_xlabel('X Coordinates (Meters)')
        ax.set_ylabel('Y Coordinates (Meters)')
        ax.set_zlabel('Z Coordinates (Meters)')
        ax.set_title(f'Uniformly Distributed Locations of the {self.commander}')
        
        
        ybottom, ytop = ax.get_ylim()
        xbottom, xtop = ax.get_xlim()

        if xtop >= ytop:
            max = xtop
        else:
            min = ytop

        if xbottom <= ybottom:
            min = xbottom
        else:
            min = ybottom

        ax.set_ylim(min, max)
        ax.set_xlim(min, max)
        

        fig.set_size_inches(10.5, 6, forward=True)
        plt.show()

    def distance_to_nuke(self, unit_coordinates_dict, nsnw_drop):
        distances = []
        for key in unit_coordinates_dict:
            for i in range(len(unit_coordinates_dict[key])):
                distances.append(math.sqrt((nsnw_drop[:,0][0] - unit_coordinates_dict[key][:, 0][i])**2 + (nsnw_drop[:,1][0] - unit_coordinates_dict[key][:, 1][i])**2 +(nsnw_drop[:,2][0] - unit_coordinates_dict[key][:, 0][i])**2))
        distances = np.array((distances))
        return distances

    def assigning_unit_type(self):
        unit_type = np.array(())
        company = np.array(())
        for co in self.CO_list:
            co_striped = self.base_unit(co)
            for key, value in self.unit_dict[co_striped].items():    
                item = np.array([co for _ in range(self.unit_dict[co_striped][key])])
                company = np.append(company,item)

        for co in self.CO_list:
            co = self.base_unit(co)
            for key, value in self.unit_dict[co].items():
                item = np.array([key for _ in range(self.unit_dict[co][key])])
                unit_type = np.append(unit_type, item)
                
        return unit_type, company

    def creating_df(self, unit_coordinates_dict, d, unit_type, company, blast, thermal):
        units_df = pd.DataFrame()
        for i in range(len(self.CO_list)):
            df2 = pd.DataFrame(unit_coordinates_dict[i])
            units_df = pd.concat([units_df, df2], ignore_index=True)
        
        units_df.columns = ['X', 'Y', 'Z']
        units_df['Distance'] = d
        units_df['Unit_Type'] = unit_type
        units_df['Company'] = company
        units_df['Blast Effects'] = blast
        units_df['Thermal Effects'] = thermal

        conditions = [
        (units_df['Blast Effects'] <= 5) & (units_df['Unit_Type'] == 'Soldier') | # Start of Blast effects
        (units_df['Blast Effects'] <= 10) & (units_df['Unit_Type'] == 'Vehicle') |
        (units_df['Blast Effects'] <= 10) & (units_df['Unit_Type'] == 'Truck') |
        (units_df['Blast Effects'] <= 1.5) & (units_df['Unit_Type'] == 'Helicopter') |
        (units_df['Blast Effects'] <= 10) & (units_df['Unit_Type'] == 'LA') |
        (units_df['Blast Effects'] <= 10) & (units_df['Unit_Type'] == 'Tank') # End of Blast effects
        ,
        (units_df['Blast Effects'] > 5) & (units_df['Blast Effects'] < 12) & (units_df['Unit_Type'] == 'Soldier') |  # Start of Blast effects
        (units_df['Blast Effects'] > 10) & (units_df['Blast Effects'] < 15) & (units_df['Unit_Type'] == 'Vehicle') |
        (units_df['Blast Effects'] > 10) & (units_df['Blast Effects'] < 15) & (units_df['Unit_Type'] == 'Truck') |
        (units_df['Blast Effects'] > 1.5) & (units_df['Blast Effects'] < 3) & (units_df['Unit_Type'] == 'Helicopter') |
        (units_df['Blast Effects'] > 10) & (units_df['Blast Effects'] < 15) & (units_df['Unit_Type'] == 'LA') |
        (units_df['Blast Effects'] > 10) & (units_df['Blast Effects'] < 15) & (units_df['Unit_Type'] == 'Tank') # End of Blast effects
        ,
        (units_df['Blast Effects'] >= 12) & (units_df['Unit_Type'] == 'Soldier') | # Start of Blast effects
        (units_df['Blast Effects'] >= 15) & (units_df['Unit_Type'] == 'Vehicle') |
        (units_df['Blast Effects'] >= 15) & (units_df['Unit_Type'] == 'Truck') |
        (units_df['Blast Effects'] >= 3) & (units_df['Unit_Type'] == 'Helicopter') |
        (units_df['Blast Effects'] >= 15) & (units_df['Unit_Type'] == 'LA') |
        (units_df['Blast Effects'] >= 15) & (units_df['Unit_Type'] == 'Tank') # End of Blast effects
        ]

        values = ['Fine', 'Injured', 'Dead']
        units_df['Status'] = np.select(conditions, values)

        return units_df

    def df_summary(self, units_df):
        #Blast Effects Plot
        units_df.plot(kind = 'scatter', x = 'Distance', y = 'Blast Effects', title = 'Scatter Plot of Blast Effects Over Distance', xlabel = 'Distance (meters)', ylabel = 'Blast Effects (PSI)')
        plt.show()

        #Thermal Effects Plot
        units_df.plot(kind = 'scatter', x = 'Distance', y = 'Thermal Effects', title = 'Scatter Plot of Thermal Effects Over Distance', xlabel = 'Distance (meters)', ylabel = 'Thermal Effects (cal/cm\u00b2)')
        plt.show()

        #Bar plot of counts
        fig = plt.figure()
        ax = fig.add_subplot(111) 
        units_df[['Status','Unit_Type','Company']].value_counts().plot(kind = 'bar', title = 'Bar Plot of Counts of Unit Status', xlabel = 'Unit Status', ylabel = 'Counts', rot = 15)
        fig.set_size_inches(10.5, 6, forward=True)

        for p in ax.patches:
            ax.annotate(str(p.get_height()), (p.get_x(), p.get_height() + .5))

        plt.show()

    def map_plot(self, units_df, nuke_df, latitude, longitude):
        
        #https://gis.stackexchange.com/questions/78838/converting-projected-coordinates-to-lat-lon-using-python
        def xy_to_latlong(x,y):
            transformer = Transformer.from_crs("epsg:3857", "epsg:4326")
            lat,lon = transformer.transform(x,y)
            return (lat,lon)

        units_df['coordinates'] = list(zip(units_df['X'], units_df['Y']))

        latlon = [xy_to_latlong(x, y) for x, y in units_df['coordinates'] ]

        units_df['latlon'] = latlon
        # Split that column out into two separate columns - latitude and longitude
        units_df[['latitude', 'longitude']] = units_df['latlon'].apply(pd.Series)

        units_df['latitude'] = units_df['latitude'].apply(lambda x: x + latitude)
        units_df['longitude'] = units_df['longitude'].apply(lambda x: x + longitude)

        #https://towardsdatascience.com/creating-an-interactive-map-in-python-using-bokeh-and-pandas-f84414536a06
        def latlong_to_mercator(lat, lon):
            r_major = 6378137.000
            x = r_major * np.radians(lon)
            scale = x/lon
            y = 180.0/np.pi * np.log(np.tan(np.pi/4.0 + 
                lat * (np.pi/180.0)/2.0)) * scale
            return (x, y)

        # Define coord as tuple (lat,long)
        units_df['coordinates'] = list(zip(units_df['latitude'], units_df['longitude']))

        # Obtain list of mercator coordinates
        mercators = [latlong_to_mercator(x, y) for x, y in units_df['coordinates'] ]

        units_df['mercator'] = mercators
        # Split that column out into two separate columns - mercator_x and mercator_y
        units_df[['mercator_x', 'mercator_y']] = units_df['mercator'].apply(pd.Series)

        units_df = units_df.drop(labels = ['mercator', 'latlon', 'coordinates'], axis = 1)
        
        # Select tile set to use
        chosentile = get_provider('OSM')

        # Tell Bokeh to use df as the source of the data
        unit_source = ColumnDataSource(data=units_df)
        nuke_source = ColumnDataSource(data=nuke_df)

        # Set tooltips - these appear when we hover over a data point in our map, very nifty and very useful
        tooltips = [("Unit Type","@Unit_Type"), ("Status","@Status"), ("Z","@Z")]

        # Create figure
        p = figure(title = 'Unit Location', 
        x_axis_type = "mercator", 
        y_axis_type = "mercator", 
        x_axis_label = 'Longitude', 
        y_axis_label = 'Latitude', 
        tooltips = tooltips)

        # Add map tile
        p.add_tile(chosentile)

        # Add points using mercator coordinates
        #Units
        p.circle(x = 'mercator_x', y = 'mercator_y', source = unit_source, size = 10, fill_alpha = 1)
        #Nuke
        p.circle(x = 'mercator_x', y = 'mercator_y', source = nuke_source, size = 10, fill_alpha = 1 , color = 'red')

        # Display in notebook
        output_notebook()
        # Save as HTML
        output_file('Wargame.html', title='Effects of a Non-Strategic Nuclear Weapon on Army Units')

        show(p)

    def plot_unit_status(self, units_df, nuke_df):
        #Plotting the circle
        
        #circle = np.array([(unit_center_x + radius * math.cos(i), unit_center_y + radius * math.sin(i), unit_center_z) for i in np.arange(0, math.pi * 2, 0.01)])

        fig = plt.figure()
        plt.clf()

        ax = fig.add_subplot(111, projection='3d')

        ax.scatter3D(nuke_df['X'], nuke_df['Y'], nuke_df['Z'])
        
        colors = {'Dead':'red', 'Injured':'yellow', 'Fine':'green'}

        groups = units_df.groupby('Status')
        for name, group in groups:
            ax.scatter3D(group['X'], group['Y'], marker = 'o', label = name, c = group['Status'].map(colors))

        ax.legend(title='Units', bbox_to_anchor=(1.05, 1), loc='upper left')
        
        ax.set_xlabel('X Coordinates (Meters)')
        ax.set_ylabel('Y Coordinates (Meters)')
        ax.set_zlabel('Z Coordinates (Meters)')
        ax.set_title(f'Statuses of the {self.commander}')
        
        '''
        ybottom, ytop = ax.get_ylim()
        xbottom, xtop = ax.get_xlim()

        if xtop >= ytop or xbottom <= ybottom:
            ax.set_ylim(xbottom, xtop)
        elif ytop > xtop or ybottom < xbottom:
            ax.set_xlim(ybottom, ytop)
        '''

        fig.set_size_inches(10.5, 6, forward=True)
        plt.show()

def monte_carlo(trials, unit, nuke, unit_center, theta):
    st = time.time()
    
    monte_carlo_df = pd.DataFrame()
    counter = 0
    for i in range(trials):
        nuke.drop_nuke()

        if counter < 1:
            nuke.plot_nuke()

        unit_size = unit.unit_size()

        if counter < 1:
            total_size = 0
            for i in range(len(unit_size)):
                total_size += int(unit_size[i])

        unit_coordinates_dict = {}
        for i in range(len(unit.CO_list)):
            key = i
            value = unit.unit_location(unit_center, 564.19, int(unit_size[i]), i, theta)
            unit_coordinates_dict[key] = value

        d = unit.distance_to_nuke(unit_coordinates_dict,nuke.nsnw_drop)

        unit_type, company = unit.assigning_unit_type()

        blast, thermal = nuke.nuke_effects(d)

        units_df = unit.creating_df(unit_coordinates_dict, d, unit_type, company, blast, thermal)
        nuke_df = nuke.creating_df()

        if counter < 1:
            unit.plot_unit(units_df ,nuke_df)

        hold_df = units_df[['Status','Unit_Type','Company']].value_counts().to_frame()
        sum = hold_df.groupby(hold_df.index).sum()
        sum = sum.transpose()
        monte_carlo_df =pd.concat([monte_carlo_df, sum])
        monte_carlo_df = monte_carlo_df.fillna(0)
        counter += 1
    print(time.time()-st)
    return monte_carlo_df, units_df, nuke_df

def monte_carlo_plots(monte_carlo_df):
    results_df = pd.DataFrame()

    for i in monte_carlo_df.columns:
        results_dict = {}
        status, units, company = i
        mean = monte_carlo_df[i][:].mean()
        mean = round(mean, 2)
        sd = monte_carlo_df[i][:].std()
        sd = round(sd, 2)
        var = monte_carlo_df[i][:].var()

        #DISCRETE
        #https://stackoverflow.com/questions/59308441/fitting-for-discrete-data-negative-binomial-poisson-geometric-distribution
        p = (mean / (var + .000001))
        r = p * mean / (1-p)

        log_likelihoods = {}

        #nbinom
        log_likelihoods['nbinom'] = monte_carlo_df[i][:].map(lambda val: nbinom.logpmf(val, r, p)).sum()

        #poisson
        lambda_ = mean
        log_likelihoods['poisson'] = monte_carlo_df[i][:].map(lambda val: poisson.logpmf(val, lambda_)).sum()

        #geometric
        p = 1 / mean
        log_likelihoods['geometric'] = monte_carlo_df[i][:].map(lambda val: geom.pmf(val, p)).sum()

        best_fit = max(log_likelihoods, key=lambda x: log_likelihoods[x])
        print("Best fit:", best_fit)
        print("log_Likelihood:", log_likelihoods[best_fit])

        #CONTINUOS
        #Finding the best distribution
        #https://medium.com/the-researchers-guide/finding-the-best-distribution-that-fits-your-data-using-pythons-fitter-library-319a5a0972e9
        data = monte_carlo_df[i].values

        distributions = ['gamma',
                         'lognorm',
                          "beta",
                          "burr",
                          "norm",
                          "uniform",
                          "t",
                          "cauchy", 
                          "chi2", 
                          "expon", 
                          "exponpow",
                          "powerlaw", 
                          "rayleigh"]

        f = Fitter(data, distributions = distributions)

        f.fit()

        distribution = f.get_best(method = 'sumsquare_error')
        key = list(distribution.keys())

        results_dict[f'{status} {units}s in {company}'] = (mean, sd, key[0])

        df1 = pd.DataFrame(distribution)
        df1 = df1.transpose()
        df1 = df1.set_index(pd.Index([f'{status} {units}s in {company}']))

        df2 = pd.DataFrame(results_dict)
        df2 = df2.transpose()
        df2.columns = ['Mean', 'Standard Deviation', 'Distribution']

        df2 = df2.merge(df1, left_index = True, right_index = True)

        results_df = pd.concat([results_df, df2], ignore_index = False)

        fig, (ax1, ax2) = plt.subplots(1, 2)

        fig.set_size_inches(10.5, 6, forward=True)

        #histogram subplot
        ax1.hist(monte_carlo_df[i][:], bins = 25)
        ax1.title.set_text(f'Histogram of {units}s That Are {status} in {company}')
        ax1.set_xlabel(f'Amount of {units}s That Are {status} in {company}')
        ax1.set_ylabel('Count of Occurances')
        ax1.set_ylim(0)

        #distribution subplot
        #https://stackoverflow.com/questions/69941784/is-there-a-library-that-will-help-me-fit-data-easily-i-found-fitter-and-i-will
        
        errorlist = sorted([[f._fitted_errors[dist], dist] for dist in distributions])[:3]

        for err, dist in errorlist:
            f.plot_pdf(names = dist, Nbest=1, lw=1, method='sumsquare_error')
        
        ax2.title.set_text(f'Distribution of {units}s That Are {status}')
        ax2.set_xlabel(f'Amount of {units}s That Are {status}')
        ax2.set_ylabel('Probability')
        ax2.set_ylim(0)

        plt.show()

    return results_df

#Defining the classes
#Nuke parameters
nuke_deployment = 'Bomb'
nuke_center = (0, 0, 213.36)
Yield = 30

commander = 'IN BN'
unit = 'IBCT'
phase = 4
individual_unit = False
trials = 1000
unit_center = (0, 0, 0)
theta = (2 * np.pi)/ 1

nuke = Nuke(nuke_deployment, nuke_center, Yield)
unit = Unit(commander, unit, phase, individual_unit)

print(nuke.nsnw_df)
print(unit.blueforce_df)

#Finding the size of each company in the unit and the total of all companies together
unit_size = unit.unit_size()

#Setting up the results df
#Moscow
#latitude = 55.726914
#longitude = 37.618830
#West Point
#latitude = 41.391368
#longitude = -73.958556
#Home
latitude = 43.104780
longitude = -88.342880

monte_carlo_df, units_df, nuke_df = monte_carlo(trials, unit, nuke, unit_center, theta)

#print(tabulate(units_df, headers='keys', tablefmt='psql'))
print(units_df)
print(nuke_df)

unit.df_summary(units_df)

nuke_df = nuke.plotting_df(nuke_df, latitude, longitude)

monte_carlo_df.columns = monte_carlo_df.columns.to_flat_index()

unit.plot_unit_status(units_df, nuke_df)

results_df = monte_carlo_plots(monte_carlo_df)

print(results_df)

#gmaps TEST

'''
import gmaps


api_key = 'AIzaSyDaxW5RqSoRgn-T68NOoL6O3Bo4C2JxF9k'

gmaps.configure(api_key=api_key)

unit_locations_layer = gmaps.symbol_layer(units_df[['latitude', 'longitude']], fill_color='green', stroke_color='green', scale=2)
fig = gmaps.figure()
fig.add_layer(unit_locations_layer)

fig

from ipywidgets.embed import embed_minimal_html
embed_minimal_html('Wargame.html', views=[fig])
'''

#pydeck TEST

'''
import pydeck as pdk

UK_ACCIDENTS_DATA = ('https://raw.githubusercontent.com/uber-common/'
                     'deck.gl-data/master/examples/3d-heatmap/heatmap-data.csv')

# Define a layer to display on a map
layer = pdk.Layer(
    'HexagonLayer',
    UK_ACCIDENTS_DATA,
    get_position=['lng', 'lat'],
    auto_highlight=True,
    elevation_scale=50,
    pickable=True,
    elevation_range=[0, 3000],
    extruded=True,                 
    coverage=1)

# Set the viewport location
view_state = pdk.ViewState(
    longitude=-1.415,
    latitude=52.2323,
    zoom=6,
    min_zoom=5,
    max_zoom=15,
    pitch=40.5,
    bearing=-27.36)

# Render
r = pdk.Deck(layers=[layer], initial_view_state=view_state)

r.to_html('demo.html', open_browser = True)
'''


#tkintermapview TEST

'''
#must pip3 install tkintermapview
import tkintermapview

# create tkinter window
root_tk = tkinter.Tk()
root_tk.geometry(f"{800}x{800}")
root_tk.title("Wargame.py")

#https://gis.stackexchange.com/questions/2951/algorithm-for-offsetting-a-latitude-longitude-by-some-amount-of-meters
#Position, decimal degrees
lat = latitude
lon = longitude

#Earth's radius, sphere
earth_r = 6378137

#offsets in meters
dnorth = 1000
deast = -1000

#Coordinate offsets in radians
dLat = dnorth/earth_r
dLon = deast/(earth_r* math.cos(math.pi * lat/180))

#OffsetPosition, decimal degrees
new_lat = lat + dLat * 180/ math.pi
new_lon = lon + dLon * 180/ math.pi 
print(new_lat)
print(new_lon)

# create map widget
map_widget = tkintermapview.TkinterMapView(root_tk, width=800, height=600, corner_radius=0)
#map_widget.set_tile_server("https://mt0.google.com/vt/lyrs=s&hl=en&x={x}&y={y}&z={z}&s=Ga", max_zoom=22)  # google satellite
map_widget.set_tile_server("https://mt0.google.com/vt/lyrs=m&hl=en&x={x}&y={y}&z={z}&s=Ga", max_zoom=22)  # google normal
map_widget.place(relx=0.5, rely=0.5, anchor=tkinter.CENTER)
map_widget.set_position(latitude, longitude)
map_widget.set_zoom(13)

marker_1 = map_widget.set_marker(latitude, longitude)
marker_2 = map_widget.set_marker(new_lat, new_lon)

root_tk.mainloop()
'''

#Bokeh TEST

from bokeh.io import output_notebook, show, output_file
from bokeh.plotting import figure, ColumnDataSource
from bokeh.tile_providers import get_provider, Vendors
from bokeh.palettes import PRGn, RdYlGn
from bokeh.transform import linear_cmap,factor_cmap
from bokeh.layouts import row, column
from bokeh.models import GeoJSONDataSource, LinearColorMapper, ColorBar, NumeralTickFormatter

#unit.map_plot(units_df, nuke_df, latitude, longitude)

'''
# Select tile set to use
chosentile = get_provider('OSM')

# Tell Bokeh to use df as the source of the data
unit_source = ColumnDataSource(data=units_df)
nuke_source = ColumnDataSource(data=nuke_df)

# Set tooltips - these appear when we hover over a data point in our map, very nifty and very useful
tooltips = [("Unit Type","@Unit_Type"), ("Status","@Status"), ("Z","@Z")]

# Create figure
p = figure(title = 'Unit Location', 
x_axis_type="mercator", 
y_axis_type="mercator", 
x_axis_label = 'Longitude', 
y_axis_label = 'Latitude', 
tooltips = tooltips)

# Add map tile
p.add_tile(chosentile)

# Add points using mercator coordinates
#Units
p.circle(x = 'mercator_x', y = 'mercator_y', source = unit_source, size = 10, fill_alpha = 1)
#Nuke
p.circle(x = 'mercator_x', y = 'mercator_y', source = nuke_source, size = 10, fill_alpha = 1 , color = 'red')

# Display in notebook
output_notebook()
# Save as HTML
output_file('Wargame.html', title='Effects of a Non-Strategic Nulear Weapon on Army Units')

show(p)
'''

'''
from tkinterhtml import HtmlFrame

import tkinter as tk

root = tk.Tk()

frame = HtmlFrame(root, horizontal_scrollbar="auto")
 
frame.set_content("Wargame.html")
'''
