/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
    num_particles = 100;

    normal_distribution<double> dist_x(x, std[0]);
    normal_distribution<double> dist_y(y, std[1]);
    normal_distribution<double> dist_theta(theta, std[2]);

    for(int i=0; i < num_particles; i++){

	Particle particle;

	particle.id = i;
	particle.x = dist_x(gen);
	particle.y = dist_y(gen);
	particle.theta = dist_theta(gen);
	particle.weight = 1.0;

	particles.push_back(particle);
    }

    is_initialized = true;

}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
    normal_distribution<double> dist_x(0, std_pos[0]);
    normal_distribution<double> dist_y(0, std_pos[1]);
    normal_distribution<double> dist_theta(0, std_pos[2]);

    for(int i=0; i < num_particles; i++){

	double theta_particle = particles[i].theta;

	// Avoid division by zero
	if (fabs(yaw_rate) < 0.00001){
	    particles[i].x += velocity*delta_t*cos(theta_particle);
	    particles[i].y += velocity*delta_t*sin(theta_particle);
        }else{
	    particles[i].x += velocity/yaw_rate * (sin(theta_particle + yaw_rate*delta_t) - sin(theta_particle));
	    particles[i].y += velocity/yaw_rate * (cos(theta_particle) - cos(theta_particle + yaw_rate*delta_t));
	    particles[i].theta += yaw_rate*delta_t;
	}

	// Adding noise to each particule
	particles[i].x += dist_x(gen);
	particles[i].y += dist_y(gen);
	particles[i].theta += dist_theta(gen);
    }

}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
/*
    dist_min = 0;

    // initialize "dist_min" to a large number. Finally, "dist_min" value is equal to the maximum distance of an observation landmark to the frame origin. This method is not robust !
    for(int i=0; i < observations.size(); i++){

	LandmarkObs obs = observations[i];

	dist_obs = dist(obs.x, obs.y, 0, 0);

	if (dist_obs > dist_min){
	    dist_min = obs.x;
	}else{
	    
	}
    }
*/
    // data association
    for(int i=0; i < observations.size(); i++){

	LandmarkObs obs = observations[i];

	// http://www.cplusplus.com/reference/limits/numeric_limits/
	double dist_min = numeric_limits<double>::max();

	// set id to -1 allow to detect error if final "map_obs_id" is equal to -1 yet. It's an absurd value.
	int map_obs_id = -1;

	for(int j=0; j < predicted.size(); j++){

	    LandmarkObs pred = predicted[j];

	    double near_distance = dist(obs.x, obs.y, pred.x, pred.y);

	    if (near_distance < dist_min){
		dist_min = near_distance;
		map_obs_id = pred.id;
	    }
	}

	observations[i].id = map_obs_id;

    }
}

void ParticleFilter::landmarksRegion(std::vector<LandmarkObs>& predictions, const Map& map_landmarks, double& p_x, double& p_y, double sensor_range){
	// Find the map landmarks that are in a range of "sensor_range" around the particle described by "p_x", "p_y". These landmarks are described like the "predictions" of the particle.

    for (int j=0; j < map_landmarks.landmark_list.size(); j++){
	    
	int ldmk_id = map_landmarks.landmark_list[j].id_i;
	float ldmk_x = map_landmarks.landmark_list[j].x_f;
	float ldmk_y = map_landmarks.landmark_list[j].y_f;

	if (dist(ldmk_x, ldmk_y, p_x, p_y) <= sensor_range){
		predictions.push_back(LandmarkObs{ldmk_id, ldmk_x, ldmk_y});
	}
    }
}

void ParticleFilter::transformeCoordinates(std::vector<LandmarkObs>& transformed_obs, const std::vector<LandmarkObs>& observations, double& p_x, double& p_y, double p_theta){
	// Convert "observations" given in the VEHICLE'S coordinate system by "transformed_obs" given in the MAP'S coordinate system.

    for (int j=0; j < observations.size(); j++){

	double tf_x = p_x + cos(p_theta)*observations[j].x - sin(p_theta)*observations[j].y;
	double tf_y = p_y + sin(p_theta)*observations[j].x + cos(p_theta)*observations[j].y;

	transformed_obs.push_back(LandmarkObs{observations[j].id, tf_x, tf_y});

    }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html

    for(int i=0; i < num_particles; i++){

	double p_x = particles[i].x;
        double p_y = particles[i].y;
	double p_theta = particles[i].theta;

	// "predictions" are the "map_landmarks" which are in the "sensor_range" raduis of the particle i.
	vector<LandmarkObs> predictions;

	landmarksRegion(predictions, map_landmarks, p_x, p_y, sensor_range);

	// "transformed_obs" are the "observations" landmarks written in the same coordinate system than the "map_landmarks" coordinate system.
	vector<LandmarkObs> transformed_obs;
	
	transformeCoordinates(transformed_obs, observations, p_x, p_y, p_theta);
	
	
	dataAssociation(predictions, transformed_obs);


	// We need now to find which "transformed_obs" landmark is associate to which "predictions" landmark, according to the result of the "dataAssociation" function, in order to update the weight of the particle i
	double sig_x = std_landmark[0];
	double sig_y = std_landmark[1];

	double gauss_norm = 1.0/(2.0*M_PI*sig_x*sig_y);

	particles[i].weight = 1.0;

	for (int j=0; j < transformed_obs.size(); j++){

	    double obs_x, obs_y;
	    double pred_x, pred_y;

	    obs_x = transformed_obs[j].x;
	    obs_y = transformed_obs[j].y; 

	    for (int k=0; k < predictions.size(); k++){

		if (predictions[k].id == transformed_obs[j].id){
		    pred_x = predictions[k].x;
		    pred_y = predictions[k].y;
		}
	    }

	    double exponent = ((obs_x - pred_x)*(obs_x - pred_x)) / (2*sig_x*sig_x) + ((obs_y - pred_y)*(obs_y - pred_y)) / (2*sig_y*sig_y);

	    particles[i].weight *= gauss_norm * exp(-exponent);

	}
    }
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

    vector<Particle> resampled_particles;

    vector<double> weights;
  
    for (int i = 0; i < num_particles; i++) {
        weights.push_back(particles[i].weight);
    }

    //int index = rand() % num_particles;
    uniform_int_distribution<int> int_dist(0, num_particles-1);
    auto index = int_dist(gen);

    // http://www.cplusplus.com/reference/algorithm/max_element/
    double max_weight = *max_element(weights.begin(), weights.end());

    //double init_beta = rand() % max_weight;
    uniform_real_distribution<double> double_dist(0.0, max_weight);

    // Implementation of resamplig code from Lesson 14, chapter 20.
    double beta = 0.0;

    for (int i = 0; i < num_particles; i++) {
        beta += double_dist(gen) * 2.0;
        while (beta > weights[index]) {
            beta -= weights[index];
	    index = (index + 1) % num_particles;
        }
        resampled_particles.push_back(particles[index]);
    }

    particles = resampled_particles;
}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
