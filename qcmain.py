#!/usr/bin/env python
import qcrndm
import qcbox
import numpy as np
import time
#import km3pipe as kp
import km3pipe.style as kp_style
import sys, argparse
from tqdm import trange
kp_style.use('km3pipe')

if __name__ == "__main__":
    # Apply arguments
    theta = 1.
    density = 1.3
    is_eta = False
    mc_steps = 2e7
    width   = 50
    height  = 50
    qfactor = 0.17
    periodic_boundary = False
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input')
    parser.add_argument('-o', '--output')
    parser.add_argument('-a', '--anglefile', action='store_true')
    parser.add_argument('-e', '--eta', action='store_true')
    parser.add_argument('-d', '--density')
    parser.add_argument('-t', '--theta')
    parser.add_argument('-n', '--nsteps')
    parser.add_argument('-w', '--width')
    parser.add_argument('-l', '--height')
    parser.add_argument('-q', '--qfactor')
    parser.add_argument('-b', '--periodicboundary', action='store_true')
    args = parser.parse_args()

    if args.theta:
        theta = float(args.theta)
    if args.eta:
        is_eta = True
    if args.density:
        density = float(args.density)
    if args.nsteps:
        mc_steps = int(args.nsteps)
    if args.width:
        width = float(args.width)
    if args.height:
        height = float(args.height)
    if args.qfactor:
        qfactor = float(args.qfactor)
    if args.periodicboundary:
        periodic_boundary = True



    medium = qcbox.Box(density,is_eta,theta,width,height,periodic_boundary,qfactor)
    if args.input:
        medium.read_particle_positions_from_file(str(args.input))
    #medium.write_particle_positions_to_file("./test.txt")
    #

    print("Density:\t{}".format(medium.density))
    print("Eta:\t{}".format(medium.density_to_eta(medium.density)))
    print("Particle Amount:\t{}".format(medium.total_particle_number))
    print("Beta-Factor:\t{}".format(medium.beta))

    medium.display(False)

    for i in trange(int(mc_steps)):
    #     #with kp.time.Timer('mc_step'):
        medium.execute_mc_step()
    #     print(" ",i,end='\r')
    #     # sys.stdout.flush()
    #     if i%1000==0:
    #         medium.display(False)
        if i%1000==0:
            medium.display(False)

    #
    #
    medium.display(True)

    if args.anglefile:
        angle_file = str(args.input)
        angle_file = angle_file[:angle_file.rfind('.')+1]
        angle_file += "ang"
        medium.display_tiling_distribution(angle_file)
    else:
        medium.display_tiling_distribution()
    if args.output:
        output_filepath = str(args.output)
        medium.write_particle_positions_to_file(output_filepath)
    input("Press Enter to continue...")
