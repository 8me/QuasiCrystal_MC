#!/usr/bin/env python

import qcrndm
import numpy as np
import math
import numba as nb
from tqdm import trange
#import scipy.constants as const
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d

class Box(object):
    def __init__(self, density=0.8, is_eta=False, theta=1./100, width=70, height=70, periodic_boundary=True,qfactor=0.17,gamma=1.61, tiling_type=0):
        #####################################
        # Simulation Properties
        #####################################
        self.periodic_boundary = periodic_boundary
        self.tiling_type = tiling_type
        #####################################
        # NVT ensemble (canonical)
        #####################################
        # V - Volume
        self.width = width
        self.height = height#(1./((1+math.sqrt(5))*0.5))*self.width
        # T - Relative Temperature
        self.theta  = theta # = k*T / epsilon
        # Particle properties
        self.colloid_radius = 1
        self.polymer_radius = qfactor*self.colloid_radius
        # N - Numerical Density
        if not is_eta:
            self.density = density
        else:
            self.density = self._eta_to_density(density)
        # Gamma factor
        self.gamma = gamma
        #####################################
        # Further Inits
        #####################################
        self.particle_movement = 2*self.colloid_radius
        # INIT
        # -> Step Counters
        self.step_number = 0
        self.accepted_steps = 0
        # -> Random Number Gen
        self.random_generator = qcrndm.Random()
        # -> Particles
        box_area = self.width*self.height
        particle_size = math.pi*(self.colloid_radius)**2
        self.total_particle_number = int(self.density*box_area/particle_size)
        self.particle_positions = np.zeros((self.total_particle_number, 2))
        self._place_particles()

        self.q_factor = self.polymer_radius/self.colloid_radius
        self.kappa_sigma = 0.9
        # -> Temperature
        self.beta = 1./(self.theta)

        # GUI
        self.main_fig, self.main_axis = plt.subplots()
        self.fig = None
        self.axis = None

        # self.axis = np.array([])
        # figure, axis = plt.subplots()
        # self.fig = np.append(self.fig, figure)
        # self.axis = np.append(self.axis, axis)
        # figure, axis =
        # self.histogram_figure = figure
        # self.histogram_axis   = axis

    def _eta_to_density(self, eta):
        q = self.polymer_radius/self.colloid_radius
        retval = 0.25*math.pi*(eta/math.pow(1+q,2))
        return retval#math.pow(q,2)*eta/math.pow(2*self.colloid_radius+2*self.polymer_radius,2)

    def density_to_eta(self, density):
        q = self.polymer_radius/self.colloid_radius
        retval = (4/math.pi)*(density*math.pow(1+q,2))
        return retval

    def _place_particles(self):
        # Threefold tiling
        x = self.colloid_radius
        y = self.colloid_radius
        line_number = 0

        triangle_distance = 0.69*(2*self.colloid_radius)*math.sqrt(math.pi/(math.sqrt(3)*self.density))
        square_distance = self.colloid_radius*math.sqrt(math.pi/self.density)
        #print(triangle_distance)
        if self.tiling_type == 0:
            for i in range(self.particle_positions.shape[0]):
                self.particle_positions[i,0] = x
                self.particle_positions[i,1] = y
                x += triangle_distance
                if x > self.width-self.colloid_radius:
                    line_number += 1
                    x = self.colloid_radius
                    if line_number%2==1:
                        x += 0.5*triangle_distance
                    y += math.sqrt(3)*0.5*triangle_distance


        #Fourfold tiling
        elif self.tiling_type == 1:
            x = self.colloid_radius
            y = self.colloid_radius
            #line_number = 0
            for i in range(self.particle_positions.shape[0]):
                self.particle_positions[i,0] = x
                self.particle_positions[i,1] = y
                x += square_distance
                if x > self.width-self.colloid_radius:
                    line_number += 1
                    x = self.colloid_radius
                    y += square_distance

        # Random
        elif self.tiling_type == 2:
            for i in range(self.particle_positions.shape[0]):
                self.particle_positions[i] = self._create_new_global_particle_position()


    def _has_particles_around(self, x, y, r):
        retval = False
        pos = self.particle_positions
        for n in range(pos.shape[0]):
            xtest = pos[n, 0]
            ytest = pos[n, 1]
            if np.hypot(x-xtest, y-ytest) <= r:
                retval = True
                break
        return retval

    def _create_new_global_particle_position(self):
        boundary_distance = 0
        if not self.periodic_boundary:
            boundary_distance = self.colloid_radius
        x = boundary_distance+self.random_generator.rand()*(self.width-2*boundary_distance)
        y = boundary_distance+self.random_generator.rand()*(self.height-2*boundary_distance)
        return np.array([x, y])

    def _create_new_particle_position(self, xold, yold):
        x = -1
        y = -1
        do_global_jump = self.random_generator.rand() < 0.2

        while not self._is_particle_in_box(x, y):
            if do_global_jump:
                [x,y] = self._create_new_global_particle_position()
            else:
                phi = self.random_generator.rand()*2*math.pi
                r = math.sqrt(self.random_generator.rand())*self.particle_movement
                x = xold+r*math.cos(phi)
                y = yold+r*math.sin(phi)
                if self.periodic_boundary:
                    if x < 0:
                        x += self.width
                    elif x > self.width:
                        x -= self.width
                    if y < 0:
                        y += self.height
                    elif y > self.height:
                        y -= self.height

        return np.array([x, y])

    def execute_mc_step(self):
        self.step_number += 1
        dblProb = 0
        ######################
        # Dynamic Env.
        ######################
        # if self.step_number%10000 == 0:
        #     self.beta += 0.3
        #     print(self.beta)
        ######################
        # select particle
        ######################
        used_particle_number = int(math.floor(self.random_generator.rand()*self.particle_positions.shape[0]))
        xold = self.particle_positions[used_particle_number, 0]
        yold = self.particle_positions[used_particle_number, 1]
        old_system_energy = calculate_potential(self.particle_positions,
                                                self.colloid_radius,
                                                self.polymer_radius,
                                                self.step_number,
                                                used_particle_number,
                                                self.width,
                                                self.height,
                                                self.periodic_boundary,
                                                self.gamma)
        self.particle_positions[used_particle_number] = self._create_new_particle_position(xold, yold)
        new_system_energy = calculate_potential(self.particle_positions,
                                                self.colloid_radius,
                                                self.polymer_radius,
                                                self.step_number,
                                                used_particle_number,
                                                self.width,
                                                self.height,
                                                self.periodic_boundary,
                                                self.gamma)

        if old_system_energy-new_system_energy > 0:
            dblProb = 1
        else:
            try:
                dblProb = math.exp(self.beta*(old_system_energy-new_system_energy))
            except OverflowError:
                dblProb = -1

        if dblProb > self.random_generator.rand():
            self.accepted_steps += 1
        else:
            self.particle_positions[used_particle_number, 0] = xold
            self.particle_positions[used_particle_number, 1] = yold


        if self.step_number%1000 == 0 and False:
            #print(self.accepted_steps)
            if self.accepted_steps > 40:
                if self.particle_movement < 2*self.colloid_radius:
                    self.particle_movement *= 1.5
            else:
                if self.particle_movement > 0.75*self.colloid_radius:
                    self.particle_movement *= 0.75

            self.accepted_steps = 0

    def _is_particle_in_box(self, x, y):
        boundary_distance = 0
        if not self.periodic_boundary and False:
            boundary_distance = self.colloid_radius
        retval = True
        retval &= x > boundary_distance
        retval &= y > boundary_distance
        retval &= x < (self.width-boundary_distance)
        retval &= y < (self.height-boundary_distance)
        return retval

    def display(self, update_additional=False, filename=None):
        self.display_lattice(filename=filename)
        if update_additional:
            if self.fig == None:
                self.fig, self.axis = plt.subplots(2, 1)
            self.display_voronoi()
            #self.display_potential()
            #self.display_reciprocal_space()
        plt.tight_layout()

    def display_lattice(self,filename=None):
        figure = self.main_fig
        ax  = self.main_axis
        ax.clear()
        x = self.particle_positions[:, 0]
        y = self.particle_positions[:, 1]
        bbox = ax.get_window_extent().transformed(figure.dpi_scale_trans.inverted())
        point_size = math.pi*(0.9*self.colloid_radius*figure.dpi*bbox.width/self.width)**2
        ax.scatter(x, y)#,s = 0.8*point_size)
        ax.set_xlim([0, self.width])
        ax.set_ylim([0, self.height])
        figure.gca().set_aspect("equal")
        if filename is not None:
            plt.savefig(filename)
        plt.draw()
        plt.pause(0.0001)

    def display_voronoi(self):
        ax = self.axis[0]
        ax.clear()
        ax.set_xlim([0, self.width])
        ax.set_ylim([0, self.height])
        vor = Voronoi(self.particle_positions)
        voronoi_plot_2d(vor,ax,show_points=True,show_vertices=False)
        #self.fig.gca().set_aspect("equal")
        plt.draw()
        plt.pause(0.0001)

    def display_potential(self):
        ax = self.axis[0,1]
        ax.clear()
        x = np.linspace(0, 6*self.colloid_radius,100)
        y = np.zeros( (len(x)) )
        ax.set_xlim([0, 6*self.colloid_radius])
        ax.set_ylim([-1, 5])
        for i in range(len(x)):
            y[i] = get_pair_potential(x[i],self.colloid_radius, self.polymer_radius, self.step_number, self.gamma)
            #y[i] *= 2
            #y[i] *= self.beta
        ax.plot(x, y)
        plt.draw()
        plt.pause(0.0001)


    def display_reciprocal_space(self):
        ax = self.axis[1,0]
        ax.clear()
        image = np.zeros((int(self.height*10), int(self.width*10)))

        for (x,y) in self.particle_positions:
            for phi in range(0,72):
                for r in range(0,int(self.colloid_radius*10)):
                    phirad = math.pi*phi*1./36
                    rtmp = r*1./10
                    tmpx = int((x+rtmp*math.cos(phirad))*10)
                    tmpy = int((y+rtmp*math.sin(phirad))*10)
                    image[tmpy,tmpx] = 1

        reciprocal_positions = np.fft.fftshift(np.fft.fft2(image))
        # reciprocal_positions = np.fft.fftshift(reciprocal_positions)
        # reciprocal_positions = np.abs(reciprocal_positions)
        #ax.set_xlim([300,700])
        #ax.set_ylim([200,400])
        ax.imshow(image,origin='lower')# np.abs(reciprocal_positions),origin='lower')#,  interpolation='nearest')
        plt.draw()
        plt.pause(0.0001)


    def _get_unit_cell_vertices(self, point_index, voronoi):
        return_list = []

        # for vertice_index in voronoi.regions[voronoi.point_region[point_index]]:
        #     list.append(vertice_index)

        list = []

        found_degeneracies = 0

        for index in range(len(voronoi.ridge_points)):
            points = voronoi.ridge_points[index]
            if points[0] == point_index or points[1] == point_index:
                if not self._is_degenerated_ridge(index,voronoi):
                    ridge_center = 0.5*(voronoi.points[points[0]]+voronoi.points[points[1]])
                    return_list.append(ridge_center)
        return np.array(return_list)

    def _is_degenerated_ridge(self, ridge_index, voronoi):
        point1 = voronoi.vertices[voronoi.ridge_vertices[ridge_index][0]]
        point2 = voronoi.vertices[voronoi.ridge_vertices[ridge_index][1]]
        distance = np.linalg.norm(point1-point2)
        return distance<self.colloid_radius

    def _get_tiling_angles(self):
        vor = Voronoi(self.particle_positions,qhull_options="QJ")
        all_angles = []
        for point_index in trange(len(self.particle_positions)):
            #print(point_index)
            unit_cell_point = self.particle_positions[point_index]
            unit_cell_vertices = self._get_unit_cell_vertices(point_index,vor)
            unit_cell_corners = len(unit_cell_vertices)
            unit_cell_angles = get_unit_cell_angles(unit_cell_point,unit_cell_vertices)
            all_angles.extend(unit_cell_angles)
        all_angles = np.array(all_angles)

        return all_angles

    def display_tiling_distribution(self, angle_filepath=None):
        ax = self.axis[1]
        ax.clear()
        tiling_data = self._get_tiling_angles()
        if angle_filepath is not None:
            with open(angle_filepath, 'w') as angle_file:
                angle_file.write("#Width:\t%f\n"%(self.width))
                angle_file.write("#Heigth:\t%f\n"%(self.height))
                angle_file.write("#Density:\t%f\n"%(self.density))
                angle_file.write("#Beta:\t%f\n"%(self.theta))
                angle_file.write("#Colloid-Radius:\t%f\n"%(self.colloid_radius))
                angle_file.write("#Polymer-Radius:\t%f\n"%(self.polymer_radius))
                angle_file.write("#Particles:\t%d\n"%(self.total_particle_number))
                for angle_value in tiling_data:
                    angle_file.write("%f\n" % (angle_value))


        #print(tiling_data)
        #print(len(tiling_data))
        y,x = np.histogram(tiling_data, bins=314)
        x = x[:-1]

        #y = y/Box._angle_to_polygon_corners(x)

        ax.plot(x,y)
        #min_fold_degree = 3
        #max_fold_degree = 8
        #amount_of_folds = np.zeros(max_fold_degree-min_fold_degree)
        #for index in range(len(angle_histogram[0])):
        #    angle = angle_histogram[1][index]
        #    amount = angle_histogram[0][index]
        #    for degree in range(min_fold_degree,max_fold_degree):
        #        lower_angle_limit = 0.5*(self._get_angle_from_fold_degree(degree) + self._get_angle_from_fold_degree(degree-1))
        #        upper_angle_limit = 0.5*(self._get_angle_from_fold_degree(degree+1) + self._get_angle_from_fold_degree(degree))
        #        if angle > lower_angle_limit and angle < upper_angle_limit:
        #            amount_of_folds[degree-min_fold_degree] += amount
        #for degree in range(min_fold_degree,max_fold_degree):
        #    amount_of_folds[degree-min_fold_degree] /= float(degree)

        #print(amount_of_folds)
        #print(np.average(angle_histogram[0]))
        ax.grid(True)
        ax.set_xlabel('Angles [rad]')
        ax.set_ylabel('Events')
        plt.draw()
        plt.pause(0.0001)

    @staticmethod
    def _angle_to_polygon_corners(angle):
        return (2*np.pi)/(np.pi-angle)

    def _get_angle_from_fold_degree(self, degree):
        return math.pi - 2*math.pi/float(degree)


    def write_particle_positions_to_file(self, filepath):
        outfile = open(filepath,'w')
        outfile.write("#Width:\t")
        outfile.write(str(self.width))
        outfile.write('\n')
        outfile.write("#Heigth:\t")
        outfile.write(str(self.height))
        outfile.write('\n')
        outfile.write("#Density:\t")
        outfile.write(str(self.density))
        outfile.write('\n')
        outfile.write("#Beta:\t")
        outfile.write(str(self.theta))
        outfile.write('\n')
        outfile.write("#Colloid-Radius:\t")
        outfile.write(str(self.colloid_radius))
        outfile.write('\n')
        outfile.write("#Polymer-Radius:\t")
        outfile.write(str(self.polymer_radius))
        outfile.write('\n')
        outfile.write("#Particles:\t")
        outfile.write(str(self.total_particle_number))
        outfile.write('\n')
        for [x,y] in self.particle_positions:
            outfile.write(str(x))
            outfile.write(',')
            outfile.write(str(y))
            outfile.write('\n')
        outfile.close()

    def read_particle_positions_from_file(self, filepath):
        with open(filepath,'r+') as infile:
            infile.readlines()
            eof = infile.tell()
            infile.seek(0,0)
            count = 0
            while(infile.tell() != eof):
                line = infile.readline()
                if line.find("#Width:") != -1:
                    self.width = float(line[line.find("\t")+1:])
                elif line.find("#Heigth:") != -1:
                    self.height = float(line[line.find("\t")+1:])
                elif line.find("#Density:") != -1:
                    self.density = float(line[line.find("\t")+1:])
                elif line.find("#Beta:") != -1:
                    self.beta = float(line[line.find("\t")+1:])
                elif line.find("#Colloid-Radius:") != -1:
                    self.colloid_radius = float(line[line.find("\t")+1:])
                    self.particle_movement = self.colloid_radius
                elif line.find("#Polymer-Radius:") != -1:
                    self.polymer_radius = float(line[line.find("\t")+1:])
                elif line.find("#Particles:") != -1:
                    self.total_particle_number = int(line[line.find("\t")+1:])
                    self.particle_positions = np.zeros((self.total_particle_number, 2))
                else:
                    x = float(line[:line.find(",")])
                    y = float(line[line.find(",")+1:])
                    self.particle_positions[count,0] = x
                    self.particle_positions[count,1] = y
                    count += 1





@nb.jit
def calculate_potential(pos, colloid_radius, polymer_radius, mc_step, used_particle_number, width, height, periodic_boundary, gamma):
    retval = 0
    i = used_particle_number

    boundary_distance = 0

    overleaps_wall = (pos[i, 0] < boundary_distance)
    overleaps_wall |= (pos[i, 0] > (width-boundary_distance))
    overleaps_wall |= (pos[i, 1] < boundary_distance)
    overleaps_wall |= (pos[i, 1] > (height-boundary_distance))
    if overleaps_wall:
        return math.inf

    for j in range(pos.shape[0]):
        if j != used_particle_number:
            deltax = math.fabs(pos[i, 0] - pos[j, 0])
            deltay = math.fabs(pos[i, 1] - pos[j, 1])
            if periodic_boundary:
                if deltax > width/2:
                    deltax = width - deltax
                if deltay > height/2:
                    deltay = height - deltay
            retval += get_pair_potential(np.hypot(deltax, deltay), colloid_radius, polymer_radius, mc_step, gamma)
    return retval

@nb.jit
def get_unit_cell_angles(unit_cell_point,unit_cell_vertices):
    unit_cell_corners = len(unit_cell_vertices)
    unit_cell_angles = []
    for i in range(unit_cell_corners-1):
        for j in range(i+1,unit_cell_corners):
            point1 = unit_cell_vertices[i]
            point2 = unit_cell_vertices[j]
            a = point1-unit_cell_point
            b = point2-unit_cell_point
            arccosInput = np.dot(a,b)/np.linalg.norm(a)/np.linalg.norm(b)
            if arccosInput >= -1 and arccosInput <= 1:
                unit_cell_angles.append(math.acos(arccosInput))

    unit_cell_angles = np.sort(unit_cell_angles,kind='mergesort')[:unit_cell_corners]
    return unit_cell_angles

@nb.jit
def get_pair_potential(r, colloid_radius, polymer_radius, mc_step, gamma):
    retval = 0
    # Helper variables
    q_factor = polymer_radius/colloid_radius
    kappa_sigma = 0.9#1./(1+q_factor)
    golden_section = (1+math.sqrt(5))/2.
    #gamma = 1.610

    # Dotera, Oshiro, Ziherl
    if False and abs(r) < 2*1.62*colloid_radius:
        retval = 1

    #Pentrose
    if False:
        if abs(r) < 2*golden_section*colloid_radius:
            retval = 1
        elif abs(r) < 2*math.pow(golden_section,2)*colloid_radius:
            retval = 0.25
        #elif abs(r) < 2*math.pow(golden_section,3)*colloid_radius:
        #    retval = 2
    # Sandbrink
    if True:
        if abs(r) >= 2*colloid_radius and abs(r) <= 2*(polymer_radius+colloid_radius):
            retval += 1
            retval -= 3*abs(r)/(4*(1+q_factor)*colloid_radius)
            retval += 0.5*math.pow(abs(r)/(2*(1+q_factor)*colloid_radius),3)
            retval *= -gamma*kappa_sigma*math.pow(1+q_factor,2)/math.exp(-kappa_sigma)
            retval /= 2*colloid_radius
        if abs(r) >= 2*colloid_radius:
            retval += (2*colloid_radius/abs(r))*math.exp(-kappa_sigma*((abs(r)/(2*colloid_radius))-1))
        else:
            retval = get_pair_potential(2*colloid_radius, colloid_radius, polymer_radius, mc_step, gamma)


    #Testpotential
    if False:
        retval = math.pow(colloid_radius/abs(r),1)

    # Core Hardening
    if abs(r) <= 2*colloid_radius:
         retval += -0.001*mc_step*(abs(r) - 2*colloid_radius)
         if retval > 10000:
             retval = math.inf

    # Catch a possible value error
    #if retval < 0:
        #retval = 0

    return retval
