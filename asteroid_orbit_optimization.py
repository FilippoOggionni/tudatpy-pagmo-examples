###############################################################################
# IMPORT STATEMENTS ###########################################################
###############################################################################
import numpy as np
import os
#from pygmo import problem, algorithm, population, gaco, maco, sort_population_mo, island
from pygmo import *

from tudatpy.io import save2txt
from tudatpy.kernel import constants
from tudatpy.kernel.interface import spice_interface
from tudatpy.kernel.simulation import environment_setup
from tudatpy.kernel.simulation import propagation_setup
from tudatpy.kernel.astro import conversion, frames


def get_itokawa_rotation_settings( ):

    pole_declination = np.deg2rad( -66.30 )
    pole_right_ascension = np.deg2rad( 90.53 )
    meridian_at_epoch = 0.0   

    initial_orientation_j2000 = frames.inertial_to_body_fixed_rotation_matrix(
	    pole_declination, pole_right_ascension, meridian_at_epoch )
    initial_orientation_eclipj2000 = np.matmul( spice_interface.compute_rotation_matrix_between_frames(
	    "ECLIPJ2000", "J2000", 0.0 ), initial_orientation_j2000 )
    np.set_printoptions(precision=100)
    print(initial_orientation_j2000)
    print(initial_orientation_eclipj2000)

    rotation_rate = np.deg2rad( 712.143 ) / constants.JULIAN_DAY

    return environment_setup.rotation_model.simple( 
	"ECLIPJ2000", "Itokawa_Fixed", initial_orientation_eclipj2000, 0.0, rotation_rate )


def get_itokawa_ephemeris_settings( itokawa_gravitational_parameter ):

    itokawa_kepler_elements = np.array([
	1.324118017407799 * constants.ASTRONOMICAL_UNIT,
       	.2801166461882852,
	np.deg2rad( 1.621303507642802 ),
	np.deg2rad( 162.8147699851312 ),
	np.deg2rad( 69.0803904880264 ),
	np.deg2rad( 187.6327516838828 ) ] )
    itokawa_kepler_elements[5] = conversion.mean_to_true_anomaly(
	eccentricity = itokawa_kepler_elements[1], 
        mean_anomaly = itokawa_kepler_elements[5] )	

    kepler_elements_reference_julian_day = 2459000.5
    kepler_elements_reference_epoch = ( 
        kepler_elements_reference_julian_day - constants.JULIAN_DAY_ON_J2000 ) * constants.JULIAN_DAY
    
    return environment_setup.ephemeris.keplerian( 
	itokawa_kepler_elements,
	kepler_elements_reference_epoch, 
	itokawa_gravitational_parameter, "Sun", "ECLIPJ2000" )

def get_itokawa_gravity_field_settings( ):

    itokawa_gravitational_parameter = 2.36
    normalized_cosine_coefficients = np.array([
    	[1.0, 0.0, 0.0, 0.0, 0.0], 
    	[-0.145216, 0.0, 0.219420, 0.0, 0.0], 
    	[0.036115, -0.028139, -0.046894, 0.069022, 0.0],
    	[0.087852, 0.034069, -0.123263, -0.030673, 0.150282]])
    normalized_sine_coefficients = np.array([
    	[0.0, 0.0, 0.0, 0.0, 0.0], 
    	[0.0, 0.0, 0.0, 0.0, 0.0], 
    	[0.0, -0.006137, -0.046894, 0.033976, 0.0],
    	[0.0, 0.004870, 0.000098, -0.015026, 0.011627]])
    return environment_setup.gravity_field.spherical_harmonic( 
	gravitational_parameter = itokawa_gravitational_parameter, 
	reference_radius = 161.915, 
        normalized_cosine_coefficients = normalized_cosine_coefficients, 
        normalized_sine_coefficients = normalized_sine_coefficients, 
        associated_reference_frame = "Itokawa_Fixed" )

def create_simulation_bodies( ):
    ###########################################################################
    # CREATE ENVIRONMENT ######################################################
    ###########################################################################

    # Create default body settings for selected celestial bodies
    bodies_to_create = ["Sun", "Earth", "Jupiter", "Saturn", "Mars"]

    # Create default body settings for bodies_to_create, with "Earth"/"J2000" as
    # global frame origin and orientation. This environment will only be valid
    # in the indicated time range
    # [simulation_start_epoch --- simulation_end_epoch]
    body_settings = environment_setup.get_default_body_settings(
        bodies_to_create,
        "SSB",
        "ECLIPJ2000")
    body_settings.add_empty_settings("Itokawa")
    itokawa_gravity_field_settings = get_itokawa_gravity_field_settings()

    body_settings.get("Itokawa").gravity_field_settings = itokawa_gravity_field_settings
    body_settings.get("Itokawa").rotation_model_settings = get_itokawa_rotation_settings()
    body_settings.get("Itokawa").ephemeris_settings = get_itokawa_ephemeris_settings(
        itokawa_gravity_field_settings.gravitational_parameter)
    body_settings.get("Itokawa").shape_settings = environment_setup.shape.spherical(162.0)

    # Create system of selected celestial bodies
    bodies = environment_setup.create_system_of_bodies(body_settings)

    ###########################################################################
    # CREATE VEHICLE ##########################################################
    ###########################################################################

    # Create vehicle objects.
    bodies.create_empty_body("Spacecraft")
    bodies.get_body("Spacecraft").set_constant_mass(400.0)

    # Create radiation pressure settings, and add to vehicle
    reference_area_radiation = 4.0
    radiation_pressure_coefficient = 1.2
    radiation_pressure_settings = environment_setup.radiation_pressure.cannonball(
        "Sun",
        reference_area_radiation,
        radiation_pressure_coefficient)
    environment_setup.add_radiation_pressure_interface(
        bodies,
        "Spacecraft",
        radiation_pressure_settings)

    return bodies

def get_acceleration_models( bodies_to_propagate,
                             central_bodies,
                             bodies ):

    # Define accelerations acting on Spacecraft by Sun and Earth.
    accelerations_settings_spacecraft = dict(
        Sun=
        [
            propagation_setup.acceleration.cannonball_radiation_pressure(),
            propagation_setup.acceleration.point_mass_gravity()
        ],
        Itokawa=
        [
            propagation_setup.acceleration.spherical_harmonic_gravity(4, 4)
        ],
        Jupiter=
        [
            propagation_setup.acceleration.point_mass_gravity()
        ],
        Saturn=
        [
            propagation_setup.acceleration.point_mass_gravity()
        ]
        # Mars=
        # [
        #     propagation_setup.acceleration.point_mass_gravity()
        # ],
        # Earth=
        # [
        #     propagation_setup.acceleration.point_mass_gravity()
        # ]
        )

    # Create global accelerations settings dictionary.
    acceleration_settings = {"Spacecraft": accelerations_settings_spacecraft}

    # Create acceleration models.
    return propagation_setup.create_acceleration_models(
        bodies,
        acceleration_settings,
        bodies_to_propagate,
        central_bodies)

def get_termination_settings(
        mission_initial_time,
        mission_duration,
        minimum_altitude,
        maximum_altitude ):

    time_termination_settings = propagation_setup.propagator.time_termination(
        mission_initial_time + mission_duration,
        terminate_exactly_on_final_condition=False
    )
    # Altitude
    upper_altitude_termination_settings = propagation_setup.propagator.dependent_variable_termination(
        dependent_variable_settings=propagation_setup.dependent_variable.relative_distance('Spacecraft', 'Itokawa'),
        limit_value=maximum_altitude,
        use_as_lower_limit=False,
        terminate_exactly_on_final_condition=False
    )
    lower_altitude_termination_settings = propagation_setup.propagator.dependent_variable_termination(
        dependent_variable_settings=propagation_setup.dependent_variable.relative_distance('Spacecraft', 'Itokawa'),
        limit_value=minimum_altitude,
        use_as_lower_limit=True,
        terminate_exactly_on_final_condition=False
    )

    # Define list of termination settings
    termination_settings_list = [time_termination_settings,
                                 upper_altitude_termination_settings,
                                 lower_altitude_termination_settings]

    return propagation_setup.propagator.hybrid_termination(termination_settings_list,
                                                                                  fulfill_single_condition=True)

def get_dependent_variables_to_save( ):

    dependent_variables_to_save = [
        propagation_setup.dependent_variable.relative_distance(
            "Spacecraft", "Itokawa"
        ),
        propagation_setup.dependent_variable.central_body_fixed_spherical_position(
            "Spacecraft", "Itokawa"
        )
    ]

    return dependent_variables_to_save


global_integrator_settings = None
global_propagator_settings = None

class AsteroidOrbitProblem:
    def __init__(self,
                 bodies,
                 integrator_settings,
                 propagator_settings):
        global global_integrator_settings, global_propagator_settings
        self.bodies_function = lambda: bodies
        self.dynamics_simulator_function = lambda: None
        global_integrator_settings = integrator_settings
        global_propagator_settings = propagator_settings

    def get_bounds(self):
        """ Define the search space """
        return ([200, 0.0, 0.0, 0.0], [2000,0.3,180,360])

    def get_nobj(self):
        return 2

    def fitness(self,
                orbit_parameters: list) -> float:
        global global_integrator_settings, global_propagator_settings
        current_bodies = self.bodies_function( )
        itokawa_gravitational_parameter = current_bodies.get_body("Itokawa").gravitational_parameter
        new_initial_state = conversion.keplerian_to_cartesian(
            gravitational_parameter=itokawa_gravitational_parameter,
            semi_major_axis=orbit_parameters[ 0 ],
            eccentricity=orbit_parameters[ 1 ],
            inclination=np.deg2rad(orbit_parameters[ 2 ] ),
            argument_of_periapsis=np.deg2rad(235.7),
            longitude_of_ascending_node=np.deg2rad(orbit_parameters[ 3 ]),
            true_anomaly=np.deg2rad(139.87) )
        global_propagator_settings.reset_initial_states(new_initial_state)

        dynamics_simulator = propagation_setup.SingleArcDynamicsSimulator(
            current_bodies, global_integrator_settings, global_propagator_settings)
        self.dynamics_simulator_function = lambda: dynamics_simulator

        dependent_variables = dynamics_simulator.dependent_variable_history
        dependent_variables_list = np.vstack( list( dependent_variables.values( ) ) )
        distance = dependent_variables_list[:, 0]
        latitudes = dependent_variables_list[:, 2]
        mean_latitude = np.mean( np.absolute( latitudes ) )
        current_fitness = 1.0 / mean_latitude

        if( np.min(distance) < 150.0 ):
            current_fitness += 1.0E4
        if(np.max(distance) > 5.0E3):
            current_fitness += 1.0E4

        return [current_fitness, np.mean(distance)]


    def get_last_run_dynamics_simulator(self):

        return self.dynamics_simulator_function()


def main():
    # Load spice kernels.
    spice_interface.load_standard_kernels()

    # Set simulation start and end epochs.
    mission_initial_time = 0.0
    mission_duration = 5.0 * 86400.0

    minimum_altitude = 150.0
    maximum_altitude = 5.0E3

    bodies = create_simulation_bodies( )

    ###########################################################################
    # CREATE ACCELERATIONS ####################################################
    ###########################################################################

    bodies_to_propagate = ["Spacecraft"]
    central_bodies = ["Itokawa"]

    # Create acceleration models.
    acceleration_models = get_acceleration_models( bodies_to_propagate, central_bodies, bodies )

    # Create numerical integrator settings.
    integrator_settings = propagation_setup.integrator.runge_kutta_variable_step_size(
        mission_initial_time, 1.0, propagation_setup.integrator.RKCoefficientSets.rkf_78,
        1.0E-6,  86400.0, 1.0E-8, 1.0E-8 )

    ###########################################################################
    # CREATE PROPAGATION SETTINGS #############################################
    ###########################################################################

    # Define list of dependent variables to save.
    dependent_variables_to_save = get_dependent_variables_to_save( )

    # Create propagation settings.
    termination_settings = get_termination_settings(
            mission_initial_time, mission_duration, minimum_altitude, maximum_altitude )

    initial_state = np.zeros(6)
    propagator_settings = propagation_setup.propagator.translational(
        central_bodies,
        acceleration_models,
        bodies_to_propagate,
        initial_state,
        termination_settings,
        output_variables=dependent_variables_to_save
    )

    ###########################################################################
    # PROPAGATE ORBIT #########################################################
    ###########################################################################

    orbitProblem = AsteroidOrbitProblem( bodies, integrator_settings, propagator_settings )

    algo = algorithm(moead(gen=1))
    prob = problem(orbitProblem)
    pop = population(prob, size=50)

    for i in range(1):
        pop = algo.evolve(pop)
        print('Current iteration')
        print(i)
        print(pop.get_f())

    dynamics_simulator = orbitProblem.get_last_run_dynamics_simulator( )
    print(dynamics_simulator.dependent_variable_history)
    problem_bounds = orbitProblem.get_bounds()
    # print(problem_bounds)
    # np.random.seed(0)
    # counter = 1
    # for iteration in range( 1000 ):
    #     semi_major_axis = np.random.uniform(problem_bounds[0][0],problem_bounds[1][0])
    #     eccentricity = np.random.uniform(problem_bounds[0][2],problem_bounds[1][1])
    #     inclination = np.random.uniform(problem_bounds[0][2],problem_bounds[1][2])
    #     node = np.random.uniform(problem_bounds[0][3],problem_bounds[1][3])
    #
    #     current_fitness = orbitProblem.fitness( [semi_major_axis,eccentricity,inclination,node])
    #     # Create simulation object and propagate dynamics.
    #     if( current_fitness < 1.0E4 ):
    #         dynamics_simulator = orbitProblem.get_last_run_dynamics_simulator( )
    #         states = dynamics_simulator.state_history
    #         dependent_variables = dynamics_simulator.dependent_variable_history
    #
    #         current_dir = '/home/dominic/Software/Tudat30Bundle/build-tudat-bundle-Desktop-Default/tudatpy/'
    #         save2txt(dependent_variables, 'dependent_variables'+str(counter)+'.dat', current_dir)
    #         save2txt(states, 'states'+str(counter)+'.dat', current_dir)
    #         counter = counter + 1
    #         print(current_fitness)
    #
    #     #print(dependent_variables)
    #     # Final statement (not required, though good practice in a __main__).
    # return 0


if __name__ == "__main__":
    main()
