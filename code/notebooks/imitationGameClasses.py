# This file includes the classes created in the 1_implementing_de_boer_2000 notebook

############################################################################################
# IMPORTS
############################################################################################

# Used for random number generation
import random as rnd
import uuid

# Used for more complex mathematical operations
import math

# Used for plotting
import matplotlib.pyplot as plt

# Used for datatype representation
import numpy as np

# Easier iterations
import itertools
from collections import Counter

# Deep copy lists
import copy

############################################################################################
# UTTERANCE
############################################################################################

class Utterance:
    """This class represent an utterance consisting of the different formants F."""
    def __init__(self, f1, f2, f3, f4):
        """Creates an utterance instance."""
        self.f1 = f1;
        self.f2 = f2;
        self.f3 = f3;
        self.f4 = f4;
        
    def print(self):
        """Prints the formants of the utterance."""
        print(f"{self.f1} {self.f2} {self.f3} {self.f4}");
        
    def string(self):
        """Returns the formants of the utterance as a long string."""
        return f"{self.f1} {self.f2} {self.f3} {self.f4}";

############################################################################################
# PHONEME
############################################################################################

class Phoneme:
    """This class represent a phoneme consisting of the different vowel parameters."""
    def __init__(self, p, h, r):
        """Creates a Phoneme instance."""
        # Values should be between 0 and 1
        self.p = max(min(p, 1), 0);
        self.h = max(min(h, 1), 0);
        self.r = max(min(r, 1), 0);
        
    def print(self):
        """Prints the vowel parameters of the phoneme."""
        print(f"{self.p} {self.h} {self.r}");

############################################################################################
# SYNTHESIZER
############################################################################################

class Synthesizer:
    """This is a class to represent utterances made by an agent."""
    def __init__(self, max_noise_ambient, max_noise_agent = 0):
        """Creates a Synthesizer instance."""
        self.max_noise_agent = max_noise_agent;
        self.max_noise_ambient = max_noise_ambient;
        
    def calculate_f1(self, phoneme: Phoneme):
        """Calculates the first formant."""
        f1 = ((-392+392*phoneme.r)*pow(phoneme.h, 2)+(596-668*phoneme.r)*phoneme.h-146+166*phoneme.r)*pow(phoneme.p, 2);
        f1 += ((348-348*phoneme.r)*pow(phoneme.h, 2)+(-494+606*phoneme.r)*phoneme.h+141-175*phoneme.r)*phoneme.p;
        f1 += ((340-72*phoneme.r)*pow(phoneme.h, 2)+(-796+108*phoneme.r)*phoneme.h+708-38*phoneme.r);
        
        return f1;


    def calculate_f2(self, phoneme: Phoneme):
        """Calculates the second formant."""
        f2 = ((-1200+1208*phoneme.r)*pow(phoneme.h, 2)+(1320-1328*phoneme.r)*phoneme.h+118-158*phoneme.r)*pow(phoneme.p, 2);
        f2 += ((1864-1488*phoneme.r)*pow(phoneme.h, 2)+(-2644+1510*phoneme.r)*phoneme.h-561+221*phoneme.r)*phoneme.p;
        f2 += ((-670+490*phoneme.r)*pow(phoneme.h, 2) + (1355-697*phoneme.r)*phoneme.h + 1517-117*phoneme.r);
        
        return f2;
    
    def calculate_f3(self, phoneme: Phoneme):
        """Calculates the third formant."""
        f3 = ((604-604*phoneme.r)*pow(phoneme.h, 2)+(1038-1178*phoneme.r)*phoneme.h+246+566*phoneme.r)*pow(phoneme.p, 2);
        f3 +=((-1150+1262*phoneme.r)*pow(phoneme.h, 2)+(-1443+1313*phoneme.r)*phoneme.h-317-483*phoneme.r)*phoneme.p;
        f3 +=((1130-836*phoneme.r)*pow(phoneme.h, 2)+(-315+44*phoneme.r)*phoneme.h+2427-127*phoneme.r);
        
        return f3;
    
    def calculate_f4(self, phoneme: Phoneme):
        """Calculates the fourth formant."""
        f4 = ((-1120+16*phoneme.r)*pow(phoneme.h, 2)+(1696-180*phoneme.r)*phoneme.h+500+522*phoneme.r)*pow(phoneme.p, 2);
        f4 +=((-140+240*phoneme.r)*pow(phoneme.h, 2)+(-578+214*phoneme.r)*phoneme.h-692-419*phoneme.r)*phoneme.p;
        f4 +=((1480-602*phoneme.r)*pow(phoneme.h, 2)+(-1220+289*phoneme.r)*phoneme.h+3678-178*phoneme.r);
        
        return f4;
    
    def synthesise(self, phoneme: Phoneme):
        """Synthesises a phoneme using the synthesiser's noise settings."""
        
        # Noise by the agent's production
        if(self.max_noise_agent > 0):
            new_p = phoneme.p + rnd.uniform(-self.max_noise_agent/2, self.max_noise_agent/2);
            new_h = phoneme.h + rnd.uniform(-self.max_noise_agent/2, self.max_noise_agent/2);
            new_r = phoneme.r + rnd.uniform(-self.max_noise_agent/2, self.max_noise_agent/2);
            
            # Make new phoneme to ensure right boundries etc
            phoneme = Phoneme(new_p, new_h, new_r)
            
        f1 = self.calculate_f1(phoneme)
        f2 = self.calculate_f2(phoneme)
        f3 = self.calculate_f3(phoneme)
        f4 = self.calculate_f4(phoneme)
        
        
        # Noise due to the communication channel
        if(self.max_noise_ambient > 0):
            f1 = f1 * (1 + rnd.uniform(-self.max_noise_ambient/2, self.max_noise_ambient/2));
            f2 = f2 * (1 + rnd.uniform(-self.max_noise_ambient/2, self.max_noise_ambient/2));
            f3 = f3 * (1 + rnd.uniform(-self.max_noise_ambient/2, self.max_noise_ambient/2));
            f4 = f4 * (1 + rnd.uniform(-self.max_noise_ambient/2, self.max_noise_ambient/2));
            
        # Make an utterance
        utterance = Utterance(f1, f2, f3, f4);
        
        return utterance; 
    
############################################################################################
# BARK OPERATOR
############################################################################################

class BarkOperator:
    """This is class is used to perform Bark operations such as distance measures.
    per default the paper used critical_distance 3.5 and second_formant_weight (lambda) 0.3 is used."""
    def __init__(self, critical_distance = 3.5, second_formant_weight = 0.3, alternative_bark_conversion = False):
        """Creates a Bark Operator instance."""
        self.critical_distance = critical_distance;
        self.second_formant_weight = second_formant_weight;
        self.better_bark_conversion = alternative_bark_conversion;

    def hertz_to_bark(self, hertz):
        """Converts hertz to bark."""
        if(self.better_bark_conversion):
            return self.hertz_to_bark_alternative(hertz);
        
        if(hertz > 271.32):
            return (math.log(hertz/271.32) / 0.1719) + 2;
        else:
            return (hertz-51)/110;

    def bark_to_hertz(self, bark):
        """Converts bark to hertz."""
        if(self.better_bark_conversion):
            return self.bark_to_hertz_alternative(bark);
        
        if(bark > 2):
            return 271.32 * math.exp( (bark-2)*0.1719 );
        else:
            return (bark*110) + 51;

    def hertz_to_bark_alternative(self, hertz):
        """Converts hertz to bark on an alternative way as used by matlab."""
        # https://nl.mathworks.com/help/audio/ref/hz2bark.html
        bark = (26.81*hertz)/(1960 + hertz) - 0.53
        
        if (bark < 2):
            bark = bark + (0.15 * (2 - bark));
            
        if (bark > 20.1):
            bark = bark + (0.22 * (bark - 20.1));
            
        return bark;

    def bark_to_hertz_alternative(self, bark):
        """Converts hertz to bark on an alternative way as used by matlab."""
        # From https://nl.mathworks.com/help/audio/ref/bark2hz.html
        if (bark < 2):
            bark = (bark - 0.3)/0.85;
            
        if (bark > 20.1):
            bark = (bark + 4.422)/1.22;
            
        hertz = 1960 * ((bark + 0.53) / (26.28 - bark));
        
        return hertz;
        
    def bark_f1(self, utterance: Utterance):
        """Converts an utterance to the first formant bark."""
        return self.hertz_to_bark(utterance.f1);
    
    def weighted_f2(self, f2_bark, f3_bark, f4_bark):
        """Calculates the effective seconf formant based on the higher frequency barks."""
        if ( f3_bark - f2_bark > self.critical_distance ):
            return ( f2_bark );
        
        # These weights are not optimal according to de Boer
        weight1 = (self.critical_distance - (f3_bark - f2_bark)) / self.critical_distance;
        weight2 = ((f4_bark - f3_bark) - (f3_bark - f2_bark)) / (f4_bark - f2_bark);
        
        if (weight2 < 0):
                weight2 = -weight2;
                
        if ((f4_bark - f2_bark) > self.critical_distance):
            return (((2 - weight1) * f2_bark) + (weight1 * f3_bark)) / 2;
        
        if ((f3_bark - f2_bark) < (f4_bark - f3_bark)):
            return (((weight2 * f2_bark) + ((2 - weight2) * f3_bark)) / 2) - 1;
        
        # Default -> there was a + between 2 and weight2 in the paper but not in the code
        return ((((2 - weight2) * f3_bark) + (weight2 * f4_bark)) / 2) - 1;
    
    def bark_f2(self, utterance: Utterance):
        """Converts and utterance to the effective second formant bark."""
        f2_bark = self.hertz_to_bark(utterance.f2)
        f3_bark = self.hertz_to_bark(utterance.f3)
        f4_bark = self.hertz_to_bark(utterance.f4)
        
        return self.weighted_f2(f2_bark, f3_bark, f4_bark);
        
    def distance_between_utterances(self, utterance_1: Utterance, utterance_2: Utterance):
        """Calculates the distance between 2 utterances."""
        f1_bark_utt1 = self.bark_f1( utterance_1 );
        f1_bark_utt2 = self.bark_f1( utterance_2 );
        
        f2_bark_utt1 = self.bark_f2( utterance_1 );
        f2_bark_utt2 = self.bark_f2( utterance_2 );
        
        f1_difference = math.pow(f1_bark_utt1 - f1_bark_utt2, 2);
        f2_difference = math.pow(f2_bark_utt1 - f2_bark_utt2, 2);
        
        return math.sqrt(f1_difference + (self.second_formant_weight * f2_difference));

    def max_merge_distance(self, noise: float):
        """Maximum merge distance for non distinct sounding utterances."""
        return (math.log(1 + noise) / 0.1719) - (math.log(1 - noise) / 0.1719);
    
############################################################################################
# SOUND
############################################################################################

class Sound:
    """This is a class used to represent known sounds in an agents repetoire."""
    def __init__(self, phoneme: Phoneme):
        """Creates a Sound instance."""
        self.phoneme = phoneme;
        self.utterance = Synthesizer(max_noise_ambient = 0).synthesise(phoneme);
        self.usage_count = 0;
        self.success_count = 0;
        
    def was_used(self):
        """Add 1 to the usage count."""
        self.usage_count += 1;
        
    def was_success(self):
        """Add 1 to the success count."""
        self.success_count += 1;
        
    def success_ratio(self):
        """Returns the success ratio."""
        if self.usage_count == 0:
            # Not used, return perfect success
            return 1;
        else:
            return self.success_count/self.usage_count;
        
    def improve(self, improved_sound):
        """Improves a Sound to the new sound by updating its phoneme and utterance."""
        self.phoneme = improved_sound.phoneme;
        self.utterance = improved_sound.utterance;
        
    def merge(self, merged_sound):
        """Merges a Sound by combining the usage and success count."""
        self.usage_count += merged_sound.usage_count
        self.success_count += merged_sound.success_count
    
############################################################################################
# AGENT
############################################################################################

class Agent:
    """This is a class used to represent agents in the experiment.
    The known_phonemes are used to represent the vowels known by the agent."""
    def __init__(self, synthesizer: Synthesizer, bark_operator: BarkOperator,
                    logger: bool = False,
                    phoneme_step_size: float = 0.1, max_similar_sound_loops: int = 20, max_semi_random_loop: int = 5,
                    sound_threshold_game: float = 0.5, sound_threshold_agent:float = 0.7, sound_minimum_tries: int = 5,
                    cleanup_prob = 0.1, new_sound_prob = 0.01, merge_prob = 1, merge_distance = 0.6):
        """Creates an instance of a Agent.
        Default settings are those from de Boer."""
        # --------- Variables to be set according to init
        # Init known sounds
        self.known_sounds = [];
        self.last_spoken_sound = None;
        self.last_heard_utterance = None;

        # Init game count variables
        self.games_count = 0;
        self.success_count = 0;
        self.speaker_count = 0;
        self.imitator_count = 0;

        # Unique name for agent
        self.name = uuid.uuid4().hex[:10].upper();

        # --------- Below parameters influence the experiment
        # Specify synthesizer and bark operator to be used
        self.synthesizer = synthesizer;
        self.bark_operator = bark_operator;

        # Whether or not to log the agents progress
        self.logger = logger;
        
        # Settings for finding similar sounds
        self.phoneme_step_size = phoneme_step_size;
        self.max_similar_sound_loops = max_similar_sound_loops;
        self.max_semi_random_loop = max_semi_random_loop;
        
        # Threshold for sound success when evaluating failed game agent himself
        self.sound_threshold_game = sound_threshold_game;
        self.sound_threshold_agent = sound_threshold_agent;
        self.sound_minimum_tries = sound_minimum_tries;
        
        # Game evaluating parameters
        self.cleanup_prob = cleanup_prob;
        self.new_sound_prob = new_sound_prob;
        self.merge_prob = merge_prob;
        self.merge_distance = merge_distance;
        
        
        
        
    def prepare_for_new_game(self, was_imitator: bool, was_succes: bool):
        """Performs actions to be taken on end of game, preparing for next game."""
        # Reset last spoken and heard parameters
        self.last_spoken_sound = None;
        self.last_heard_utterance = None;
        
        # Register played game
        self.games_count += 1;

        if was_succes:
            self.success_count += 1;

        if was_imitator:
            self.imitator_count += 1;
        else:
            self.speaker_count += 1;
        
        # Periodically cleanup sounds by throwing away bad ones
        if (rnd.uniform(0, 1) < self.cleanup_prob):
            self.remove_bad_sounds();
        
        # Periodically cleanup sounds by merging
        if (rnd.uniform(0, 1) < self.merge_prob):
            self.merge_similar_sound();
        
        # Periodically add new sounds
        if (rnd.uniform(0, 1) < self.new_sound_prob):
            self.add_semi_random_known_sound();

    def success_ratio(self):
        """Returns the success ratio of the agent in games."""
        return self.success_count / self.games_count;

    def energy(self):
        """Returns the energy the agent's sound repetoire according to its bark operator."""
        energy = 0;

        for current_sound in self.known_sounds:
            for compare_sound in self.known_sounds:
                distance = self.bark_operator.distance_between_utterances(current_sound.utterance, compare_sound.utterance);
                if distance == 0:
                    # Skip equal sounds
                    continue;
                energy += 1 / (distance ** 2);

        return energy;

    def remove_bad_sounds(self):
        """Cleans up an agent by removing sounds under threshold."""
        # Keep track of sounds needing removing
        sounds_to_remove = [];
        
        # If sound is used and below threshold - remove 
        for sound in self.known_sounds:
            if (sound.usage_count > self.sound_minimum_tries and sound.success_ratio() < self.sound_threshold_agent):
                sounds_to_remove.append(sound);
                if self.logger:
                    print(self.name + ": Removed sound during cleanup.");
                    
        # Do the remove at the end to ensure no buggy loops, ensure no dupes in list
        sounds_to_remove = list(set(sounds_to_remove))
        for sound in sounds_to_remove:
            # Keep unique values
            self.known_sounds.remove(sound);

    def merge_similar_sound(self):
        """Cleans up an agent by removing similar sounds.
        Does this by comparing the sounds using the logic from de Boer (2000) comparison code."""
        # Keep track of sounds needing removing
        sounds_to_remove = [];
                    
        # If sounds are close together - merge them by keeping "best"
        for eval_index in range(len(self.known_sounds)):
             evaluation_sound = self.known_sounds[eval_index];
             if evaluation_sound in sounds_to_remove:
                 # Don't consider this sound
                 continue;
             
             for potential_merge_index in range(eval_index + 1, len(self.known_sounds)):
                potential_merge_sound = self.known_sounds[potential_merge_index];
                if potential_merge_sound in sounds_to_remove:
                    # Don't consider this sound
                    continue;
                 
                # If phonemes to close to be confused
                phoneme_distance = math.sqrt( (evaluation_sound.phoneme.p - potential_merge_sound.phoneme.p)**2 
                                                + (evaluation_sound.phoneme.h - potential_merge_sound.phoneme.h)**2 
                                                + (evaluation_sound.phoneme.r - potential_merge_sound.phoneme.r)**2 );

                utterance_distance = self.bark_operator.distance_between_utterances(evaluation_sound.utterance, potential_merge_sound.utterance);

                
                # Numbers from de Boer
                should_merge = (phoneme_distance < 0.17 or utterance_distance < self.bark_operator.max_merge_distance(self.synthesizer.max_noise_ambient));
                if (should_merge):
                    # Determine worst and best sound
                    worst_sound = evaluation_sound if evaluation_sound.success_ratio() < potential_merge_sound.success_ratio() else potential_merge_sound
                    best_sound = evaluation_sound if evaluation_sound.success_ratio() > potential_merge_sound.success_ratio() else potential_merge_sound
                     
                    # Remove worst sound
                    sounds_to_remove.append(worst_sound);
                     
                    # Merge worst sound to best sound
                    best_sound.merge(worst_sound);
        
        # Do the remove at the end to ensure no buggy loops, ensure no dupes in list
        sounds_to_remove = list(set(sounds_to_remove))
        for sound in sounds_to_remove:
            self.known_sounds.remove(sound);
        
    def add_random_known_sound(self):
        """Adds random sound to agents repetoire."""
        # Create random phoneme
        new_p = rnd.uniform(0, 1);
        new_h = rnd.uniform(0, 1);
        new_r = rnd.uniform(0, 1);
        phoneme = Phoneme(new_p, new_h, new_r);
        
        # Add phoneme to known sounds
        sound = Sound(phoneme);
        self.known_sounds.append(sound);
        
        if self.logger:
            print(self.name + ": Added a random sound to my repetoire.");
        
    def add_semi_random_known_sound(self):
        """Adds random sound to agents repetoire by trying max_semi_random_loop variants.
        The variant with the highest summed distance to other vowels is picked."""
        # First pick a random one and assign it a best
        new_p = rnd.uniform(0, 1);
        new_h = rnd.uniform(0, 1);
        new_r = rnd.uniform(0, 1);
        phoneme = Phoneme(new_p, new_h, new_r);
        best_sound = Sound(phoneme);
        best_distance = 0;
        for old_sound in self.known_sounds:
                best_distance += self.bark_operator.distance_between_utterances(best_sound.utterance, old_sound.utterance);
        
        # Now try the remainder
        for i in range(self.max_semi_random_loop - 1):
            # Create random sound
            new_p = rnd.uniform(0, 1);
            new_h = rnd.uniform(0, 1);
            new_r = rnd.uniform(0, 1);
            phoneme = Phoneme(new_p, new_h, new_r);
            new_sound = Sound(phoneme);

            # calculate distance
            distance = 0;
            for old_sound in self.known_sounds:
                distance += self.bark_operator.distance_between_utterances(new_sound.utterance, old_sound.utterance);

            # Check if best distance
            if distance > best_distance:
                best_distance = distance;
                best_sound = new_sound;

        # Add semi random sound
        self.known_sounds.append(best_sound);

        if self.logger:
            print(self.name + ": Added a semi random sound to my repetoire.");
        
    def improve_sound(self, original_sound: Sound, goal_utterance: Utterance):
        """Returns improved original sound which is more like the goal sound.
        Considers all permutations of phoneme using phoneme_step_size"""
        # Determine all possible variations of parameter modifications
        variations = [p for p in itertools.product([-self.phoneme_step_size, 0, self.phoneme_step_size], repeat=3)];
        
        # Init vars
        best_distance = float('inf');
        best_sound = original_sound;
        for variation in variations:
            # Create variation sound
            new_p = original_sound.phoneme.p + variation[0];
            new_h = original_sound.phoneme.h + variation[1];
            new_r = original_sound.phoneme.r + variation[2];
            new_phoneme = Phoneme(new_p, new_h, new_r);
            new_sound = Sound(new_phoneme);
            
            # Test variation
            new_distance = self.bark_operator.distance_between_utterances(goal_utterance, new_sound.utterance);
            if (new_distance < best_distance):
                best_distance = new_distance;
                best_sound = new_sound;
            
        # Return best found variation
        return best_sound;
        
        
    def add_similar_sound(self, goal_utterance: Utterance):
        """Adds sound to agents repetoire that sounds similar to the given utterance."""
        # Start from a 'corner' as per de Boer's code
        best_distance = float('inf');
        best_sound = None;
        for i in range(8):
            new_p = (i % 2)*0.5+0.25;
            new_h = ((i /2) % 2)*0.5+0.25;
            new_r = (i / 4)*0.5+0.25;
            new_phoneme = Phoneme(new_p, new_h, new_r);
            new_sound = Sound(new_phoneme);
            
            # Test variation
            new_distance = self.bark_operator.distance_between_utterances(goal_utterance, new_sound.utterance);
            if (new_distance < best_distance):
                best_distance = new_distance;
                best_sound = new_sound;
        
        # Improve sound for specified amount of times
        for i in range(self.max_similar_sound_loops):
            best_sound = self.improve_sound(best_sound, goal_utterance);
            
        if self.logger:
            print(self.name + ": Added a similar sound to the one I heard to my repetoire.");
            
        # Add the best sound
        self.known_sounds.append(best_sound);
        

    def find_similar_sound(self, goal_utterance: Utterance):
        """Returns sound in repetoire closes to given utterance."""
        best_distance = float('inf');
        best_sound = None;
        
        for sound in self.known_sounds:
            new_distance = self.bark_operator.distance_between_utterances(goal_utterance, sound.utterance);
            if (new_distance < best_distance):
                best_distance = new_distance;
                best_sound = sound;
                
        return best_sound;
        
    def say_something(self):
        """Produces a random utterance and stores it has said it.
        Adds a phoneme to the agents repetoire if needed. """
        # Agent knows no sounds, add one
        if not self.known_sounds:
            self.add_random_known_sound();
            
        # Chose a random known phoneme
        self.last_spoken_sound = rnd.randrange(len(self.known_sounds));
        sound = self.known_sounds[self.last_spoken_sound];
        
        # Register use
        sound.was_used();
        
        # Produce an utterance from the chosen sound
        utterance = self.synthesizer.synthesise(sound.phoneme);
        
        if self.logger:
            print(self.name + ": saying " + utterance.string());
        
        # Return the utterance
        return utterance;

    def imitate_sound(self, heard_utterance: Utterance):
        """Produces an utterance based on the utterance it just heard."""
        if self.logger:
            print(self.name + ": heard " + heard_utterance.string());
            
        # Safe just heard sound
        self.last_heard_utterance = heard_utterance;
            
        # Agent knows no sounds, add one
        if not self.known_sounds:
            self.add_similar_sound(heard_utterance);
            
        # Find closest sound
        closest_sound = self.find_similar_sound(heard_utterance);
        
        # Register use
        self.last_spoken_sound = self.known_sounds.index(closest_sound);
        closest_sound.was_used();
        
        if self.logger:
            print(self.name + ": imitated " + closest_sound.utterance.string());
        
        
        # Return the utterance
        return closest_sound.utterance;
    
    def validate_imitation(self, heard_utterance: Utterance):
        """Returns true if imitation is correct according to agent, ending the game cycle."""
        if self.logger:
            print(self.name + ": heard " + heard_utterance.string());
            
        # Find closest sound
        closest_sound = self.find_similar_sound(heard_utterance);
        
        # Closest sound is sound
        good_imitation = closest_sound == self.known_sounds[self.last_spoken_sound];
        
        if good_imitation:
            self.known_sounds[self.last_spoken_sound].was_success();
        
        if self.logger:
            if good_imitation:
                print(self.name + ": confirmed match with " + closest_sound.utterance.string());
            else:
                print(self.name + ": rejected match with  " + closest_sound.utterance.string());
                
        # End of current game
        self.prepare_for_new_game(was_imitator=False, was_succes= good_imitation);
        
        return good_imitation;
    
    def process_non_verbal_imitation_confirmation(self, was_success):
        """Processes the non verbal confirmation if an imitation was correct, ending the game cycle."""
        if was_success:
            # Save success
            self.known_sounds[self.last_spoken_sound].was_success();
            # "Shift closer"
            improved_sound = self.improve_sound(self.known_sounds[self.last_spoken_sound], self.last_heard_utterance);
            self.known_sounds[self.last_spoken_sound].improve(improved_sound);
        else:
            if self.known_sounds[self.last_spoken_sound].success_ratio() < self.sound_threshold_game:
                # Probably bad sound - "Shift closer"
                improved_sound = self.improve_sound(self.known_sounds[self.last_spoken_sound], self.last_heard_utterance);
                self.known_sounds[self.last_spoken_sound].improve(improved_sound);
            else:
                # Probably good sound - add new sound to repetoire
                self.add_similar_sound(self.last_heard_utterance);
            
            
        if self.logger:
            if was_success:
                print(self.name + ": had a confirmed match, changed my sound to match closer.");
                
        # End of current game
        self.prepare_for_new_game(was_imitator=True, was_succes= was_success);
        

############################################################################################
# GAME STATE
############################################################################################

class GameState:
    """This is a class used to represent the state of a game."""
    def __init__(self, agents: list, iteration: int):
        """Creates a Game Engine instance.
        - agents: list of agent objects to be stored
        - iteration: iteration count at which this game state was captured"""
        self.agents = copy.deepcopy(agents);
        self.iteration = iteration;

    def plot(self):
        # Change plot size and color, then start new plot 
        plt.rcParams["figure.figsize"] = (10,10);
        plt.rcParams['figure.facecolor'] = 'white';
        plt.figure();
        
        # Plot the utterances on an agent per agent basis
        for agent in self.agents:
            f1 = [agent.bark_operator.bark_f1(sound.utterance) for sound in agent.known_sounds];
            f2 = [agent.bark_operator.bark_f2(sound.utterance) for sound in agent.known_sounds];
            plt.plot(f2, f1, 'o', label=agent.name);

        # Set titles
        plt.title(str(self.iteration) + " games");
        plt.xlabel("F'2 in bark");
        plt.ylabel("F1 in bark");
        
        # Change pot parameters
        plt.ylim(1, 8);
        plt.xlim(7, 16);
        plt.gca().invert_xaxis();
        plt.gca().invert_yaxis();
        plt.grid();

        # Show legend
        plt.legend(title="Agent names");

        # Reset figure size for next figures
        plt.rcParams["figure.figsize"] = plt.rcParamsDefault["figure.figsize"];
        plt.rcParams["figure.facecolor"] = plt.rcParamsDefault["figure.facecolor"];

############################################################################################
# GAME ENGINE
############################################################################################

class GameEngine:
    """This is a class used to represent an imitation game egine."""
    def __init__(self, number_of_agents: int, iterations: int, synthesizer: Synthesizer, bark_operator: BarkOperator, 
                    agent_phoneme_step_size: float = 0.1, agent_sound_threshold_game: float = 0.5, agent_sound_threshold_self:float = 0.7,
                    agent_sound_minimum_tries: int = 5, agent_new_sound_probability: float = 0.01):
        """Creates a Game Engine instance.
        - number_of_agents: number of equally loaded agents to be created, should be multiple of two
        - iterations: amount of iterations the game should be played for
        - synthesizer: synthesizer that should be used by all agents
        - bark_operator: bark operator that should be used by all agents"""
        self.number_of_agents = number_of_agents;
        self.iterations = iterations;
        self.synthesizer = synthesizer;
        self.bark_operator = bark_operator;

        # Create the agents
        self.agents = [Agent(synthesizer= synthesizer, bark_operator= bark_operator, 
                                phoneme_step_size= agent_phoneme_step_size,
                                sound_threshold_game= agent_sound_threshold_game,
                                sound_threshold_agent= agent_sound_threshold_self,
                                sound_minimum_tries= agent_sound_minimum_tries,
                                new_sound_prob = agent_new_sound_probability)
                                    for n in range(number_of_agents)];

    def __play_all_agents_imitation_round(self):
        """Plays an imitation game round where each agent is either a speaker or imitator at random."""
        # Create index list of agents and shuffle it
        agents_index_list = [x for x in range(self.number_of_agents)];
        rnd.shuffle(agents_index_list);

        # Split agent index list in two to create speakers and imitators
        speaker_agents_indexes = agents_index_list[:int(self.number_of_agents/2)];
        imitator_agents_indexes = agents_index_list[int(self.number_of_agents/2):];

        # Loop over each speaker and imitator pair
        for i in range(len(speaker_agents_indexes)):
            # Get speaker and imitator
            speaker_agent = self.agents[speaker_agents_indexes[i]];
            imitator_agent = self.agents[imitator_agents_indexes[i]];

            # Play the game
            start_utterance = speaker_agent.say_something();
            imitated_utterance = imitator_agent.imitate_sound(start_utterance);
            validation = speaker_agent.validate_imitation(imitated_utterance);
            imitator_agent.process_non_verbal_imitation_confirmation(validation);

    def __play_single_pair_imitation_round(self):
        """Plays an imitation game round where only one pair of speaker and imitator is chosen at random."""
        # chose random speaker and imitator
        speaker, imitator = rnd.sample(self.agents, 2);

        # play game
        start_utterance = speaker.say_something();
        imitated_utterance = imitator.imitate_sound(start_utterance);
        validation = speaker.validate_imitation(imitated_utterance);
        imitator.process_non_verbal_imitation_confirmation(validation);
        
    def play_imitation_game(self, checkpoints: list):
        """Plays an imitation game and returns a vector of GameState objects.
        - checkpoints: list of iteration numbers at which the state of the game should be saved (after playing that iteration)."""
        
        game_states = [None] * len(checkpoints);

        for i in range(self.iterations):
            # Play one iteration of the game
            self.__play_single_pair_imitation_round();

            # After playing the game, check if checkpoint reached for storing
            if i + 1 in checkpoints:
                # Store imitation game state
                game_states[checkpoints.index(i + 1)] = GameState(self.agents, i + 1);

        # Return the game states
        return game_states;


############################################################################################
# Statistics
############################################################################################

class Statistics:
    """This is a class used to calculate and plot some of the experiment statistics."""
    def __init__(self, bark_operator: BarkOperator):
        self.bark_operator = bark_operator;

    def sound_sizes_from_game_state(self, game_state: GameState):
        """Returns the vowel sizes of agents for the provided gamestate."""
        sound_sizes = [len(agent.known_sounds) for agent in game_state.agents];

        return sound_sizes;

    def average_agent_sound_size(self, game_states: list):
        """Returns the average agent vowel size together with the standard deviation [avg, std] for the provided list of gamestates.
        Does this one a Game State per Game State basis."""
        average_sound_sizes = [];
        
        for game_state in game_states:
            average_sound_sizes += [np.array(self.sound_sizes_from_game_state(game_state)).mean()];

        # Go to np array from sound sizes
        average_sound_sizes = np.array(average_sound_sizes);

        # return mean and std
        return [average_sound_sizes.mean(), average_sound_sizes.std()];

    def plot_agent_sound_size_distribution(self, game_states: list, right_limit: int = 10):
        """Plots a histogram of the agent's vowel sizes for the provided list of gamestates."""
        average_sound_sizes = [];
        
        for game_state in game_states:
            average_sound_sizes += [np.array(self.sound_sizes_from_game_state(game_state)).mean()];

        n_bins = len(Counter(average_sound_sizes).keys());

        # Change plot size and color, then start new plot 
        plt.rcParams["figure.figsize"] = (10,10);
        plt.rcParams['figure.facecolor'] = 'white';
        plt.figure();

        # Make histogram
        plt.hist(average_sound_sizes);
        
        # Set titles
        plt.title("Distribution for known sounds of agents");
        plt.xlabel("Repetoire size");
        plt.ylabel("Agent count");

        # Set Xlim
        plt.xlim(0, right_limit);

        # Reset figure size for next figures
        plt.rcParams["figure.figsize"] = plt.rcParamsDefault["figure.figsize"];
        plt.rcParams["figure.facecolor"] = plt.rcParamsDefault["figure.facecolor"]; 

    def success_ratios_from_agents(self, game_state: GameState):
        """Returns the vowel sizes of agents for the provided gamestate."""
        success_ratios = [agent.success_ratio() for agent in game_state.agents];

        return success_ratios;

    def average_agent_success_ratio(self, game_states: list):
        """Returns the average agent success ratio together with the standard deviation [avg, std] for the provided list of gamestates.
        Does this one a Game State per Game State basis."""
        average_success_ratios = [];
        
        for game_state in game_states:
            average_success_ratios += [np.array(self.success_ratios_from_agents(game_state)).mean()];

        # Go to np array from sound sizes
        average_success_ratios = np.array(average_success_ratios);

        # return mean and std
        return [average_success_ratios.mean(), average_success_ratios.std()];

    def plot_agent_success_ratio_distribution(self, game_states: list, left_limit: float = 0.8, n_bins: int = 10):
        """Plots a histogram of the agent's success ratios for the provided list of gamestates."""
        average_success_ratios = [];
        
        for game_state in game_states:
            average_success_ratios += [np.array(self.success_ratios_from_agents(game_state)).mean()];

        if len(Counter(average_success_ratios).keys()) < 10:
            n_bins = len(Counter(average_success_ratios).keys());

        # Change plot size and color, then start new plot 
        plt.rcParams["figure.figsize"] = (10,10);
        plt.rcParams['figure.facecolor'] = 'white';
        plt.figure();

        # Make histogram
        plt.hist(average_success_ratios, bins=n_bins);
        
        # Set titles
        plt.title("Distribution for success ratio of agents");
        plt.xlabel("Success ratio");
        plt.ylabel("Agent count");

        # Set Xlim
        plt.xlim(left_limit, 1);

        # Reset figure size for next figures
        plt.rcParams["figure.figsize"] = plt.rcParamsDefault["figure.figsize"];
        plt.rcParams["figure.facecolor"] = plt.rcParamsDefault["figure.facecolor"];

    def energy_from_agents(self, game_state: GameState):
        """Returns the energy of agents in the given game states."""
        energies = [agent.energy() for agent in game_state.agents];

        return energies;

    def average_agent_energy(self, game_states: list):
        """Returns the average agent energy together with the standard deviation [avg, std] for the provided list of gamestates.
        Does this one a Game State per Game State basis."""
        average_energies = [];
            
        for game_state in game_states:
            average_energies += [np.array(self.energy_from_agents(game_state)).mean()];
        
        # Go to np array from sound sizes
        average_energies = np.array(average_energies);

        # return mean and std
        return [average_energies.mean(), average_energies.std()];

    def plot_agent_energy_distribution(self, game_states: list, n_bins: int = 10):
        """Plots a histogram of the agent's success ratios for the provided list of gamestates."""
        average_energies = [];
            
        for game_state in game_states:
            average_energies += [np.array(self.energy_from_agents(game_state)).mean()];

        if len(Counter(average_energies).keys()) < 10:
            n_bins = len(Counter(average_energies).keys());

        # Change plot size and color, then start new plot 
        plt.rcParams["figure.figsize"] = (10,10);
        plt.rcParams['figure.facecolor'] = 'white';
        plt.figure();

        # Make histogram
        plt.hist(average_energies, bins=n_bins);
        
        # Set titles
        plt.title("Distribution for energy of agents");
        plt.xlabel("Energy");
        plt.ylabel("Agent count");

        # Reset figure size for next figures
        plt.rcParams["figure.figsize"] = plt.rcParamsDefault["figure.figsize"];
        plt.rcParams["figure.facecolor"] = plt.rcParamsDefault["figure.facecolor"];

    def plot_known_vowels_over_sounds(self, game_state: GameState):
        # Change plot size and color, then start new plot 
        plt.rcParams["figure.figsize"] = (10,10);
        plt.rcParams['figure.facecolor'] = 'white';
        plt.figure();
        
        # Init vars
        f1 = [];
        f2 = [];
        
        # Plot the utterances of the agents
        for agent in game_state.agents:
            f1 += [agent.bark_operator.bark_f1(sound.utterance) for sound in agent.known_sounds];
            f2 += [agent.bark_operator.bark_f2(sound.utterance) for sound in agent.known_sounds];
        
        plt.plot(f2, f1, 'bo', alpha=0.4, label="Agent sound");

        # Plot the known vowels
        argument_sets = [[0, 0, 0],
                    [0, 0, 1],
                    [0.5, 0, 0],
                    [0.5, 0, 1],
                    [1, 0, 0],
                    [1, 0, 1],
                    [0, 0.5, 0],
                    [0, 0.5, 1],
                    [0.5, 0.5, 0],
                    [0.5, 0.5, 1],
                    [1, 0.5, 0],
                    [1, 0.5, 1],
                    [0, 1, 0],
                    [0, 1, 1],
                    [0.5, 1, 0],
                    [0.5, 1, 1],
                    [1, 1, 0],
                    [1, 1, 1]];

        vowels = ["[a]", "[œ]", "[ɐ]", "[ɐ̹]", "[ɑ]", "[ɒ]", "[e]", "[ø]", "[ə]", "[e]", "[ɤ]", "[o]", "[i]", "[y]", "[ɨ]", "[ʉ]", "[ɯ]", "[u]"];

        synthesizer = Synthesizer(max_noise_ambient = 0);

        for i in range(0, len(argument_sets)):
            phoneme = Phoneme(argument_sets[i][0], argument_sets[i][1], argument_sets[i][2]);    
            utterance = synthesizer.synthesise(phoneme);
            vowel = vowels[i];
            f1 = self.bark_operator.bark_f1(utterance);
            f2 = self.bark_operator.bark_f2(utterance);
            plt.text(f2, f1, vowel, color='red', fontsize=18, ha='center', va='center')


        # Set titles
        plt.title("Known vowels compared to sounds of agents");
        plt.xlabel("F'2 in bark");
        plt.ylabel("F1 in bark");
        
        # Change pot parameters
        plt.ylim(1, 8);
        plt.xlim(7, 16);
        plt.gca().invert_xaxis();
        plt.gca().invert_yaxis();
        plt.grid();

        # Reset figure size for next figures
        plt.rcParams["figure.figsize"] = plt.rcParamsDefault["figure.figsize"];
        plt.rcParams["figure.facecolor"] = plt.rcParamsDefault["figure.facecolor"];