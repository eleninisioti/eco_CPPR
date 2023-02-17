import os
import sys

sys.path.append(os.getcwd())


from reproduce_CPPR_continuous.gridworld import Gridworld

from reproduce_CPPR_continuous.utils import VideoWriter
import jax
import jax.numpy as jnp
from jax import random
import matplotlib.pyplot as plt
import numpy as np
import pickle
import datetime
from evojax.util import save_model, load_model
import yaml
from evojax.policy.base import PolicyState
from evojax.task.base import TaskState
from flax.struct import dataclass



X_pos=jnp.expand_dims(jnp.arange(400),-1)
def update_log(ind,state,lx,ly,lfx,lrewards,lalive,spx,spfx,lfood):
    
    posx=state.agents.posx
    posy=state.agents.posy
    
    rewards=state.rewards
    alive=state.agents.alive
    grid=state.state
    
    avgx=(posx*alive).sum()/(alive.sum()+1e-10)
    avgy=(posy*alive).sum()/(alive.sum()+1e-10)

    avg_food_x=(X_pos*grid[:, :, 1]).sum()/(grid[:,:,1].sum())

    avg_rewards=(rewards*alive).sum()/(alive.sum()+1e-10)
    
    
    
    lx=lx.at[ind].set(avgx)
    lfx=lfx.at[ind].set(avg_food_x)
    lrewards=lrewards.at[ind].set(avg_rewards)
    lalive=lalive.at[ind].set(alive.sum())
    
    
    density=jnp.zeros((8))
    density=density.at[0].set(((posx<50)*alive).sum()/1000)
    density=density.at[1].set((jnp.logical_and(posx>50 , posx<100)*alive).sum()/1000)
    density=density.at[2].set((jnp.logical_and(posx>100 ,posx<150)*alive).sum()/1000)
    density=density.at[3].set((jnp.logical_and(posx>150 ,posx<200)*alive).sum()/1000)
    density=density.at[4].set((jnp.logical_and(posx>200 , posx<250)*alive).sum()/1000)
    density=density.at[5].set((jnp.logical_and(posx>250, posx<300)*alive).sum()/1000)
    density=density.at[6].set((jnp.logical_and(posx>300 , posx<350)*alive).sum()/1000)
    density=density.at[7].set((jnp.logical_and(posx>350 , posx<400)*alive).sum()/1000)
    spx=spx.at[:,ind].set(density)
    
    food=grid[:,:,1]
    
    density=jnp.zeros((8))
    density=density.at[0].set(food[:50].sum())
    density=density.at[1].set(food[50:100].sum())
    density=density.at[2].set(food[100:150].sum())
    density=density.at[3].set(food[150:200].sum())
    density=density.at[4].set(food[200:250].sum())
    density=density.at[5].set(food[250:300].sum())
    density=density.at[6].set(food[300:350].sum())
    density=density.at[7].set(food[350:400].sum())
    spfx=spfx.at[:,ind].set(density)
    
    
    lfood=lfood.at[ind].set(food.sum())
    
    
    return lx,ly,lfx,lrewards,lalive,spx,spfx,lfood
update_log_fn=jax.jit(update_log)

def update_movement(ind,state,prev_state,l_movement,l_valid):
    
    #movement=jnp.sqrt((state.agents.posx-prev_state.agents.posx)**2+(state.agents.posy-prev_state.agents.posy)**2)
    movement=jnp.abs(state.agents.posx-prev_state.agents.posx)+jnp.abs(state.agents.posy-prev_state.agents.posy)
    l_valid=l_valid.at[ind].set(state.agents.alive*prev_state.agents.alive*((state.agents.params-prev_state.agents.params).sum(axis=1)==0)*ind)
    
    l_movement=l_movement.at[ind].set(movement)
    print()
    return l_movement,l_valid

update_movement_fn=jax.jit(update_movement)



def compute_death_offspring(time_alive,alive,last_alive,nb_food,nb_offspring,sum_time_alive,nb_death,total_offspring,sum_nb_food,sum_nb_offspring):
    
    sum_time_alive=sum_time_alive+(time_alive*((last_alive-alive)>0)).sum()
    
    nb_death=nb_death+((last_alive-alive)>0).sum()
    
    total_offspring=total_offspring+((last_alive-alive)<0).sum()
    
    sum_nb_food=sum_nb_food+(nb_food*((last_alive-alive)>0)).sum()
    sum_nb_offspring=sum_nb_offspring+(nb_offspring*((last_alive-alive)>0)).sum()
    
    
    
    return sum_time_alive,nb_death,total_offspring,sum_nb_food,sum_nb_offspring


compute_death_offspring_fn=jax.jit(compute_death_offspring)

def update_life(ind,sum_time_alive,nb_death,total_offspring,sum_nb_food,sum_nb_offspring,l_expectancy,l_death,l_offspring,l_offspring_per,l_food_cons_per):
    l_death=l_death.at[ind].set(nb_death)
    l_offspring=l_offspring.at[ind].set(total_offspring)
    
    l_expectancy=l_expectancy.at[ind].set(sum_time_alive/nb_death)
    
    l_offspring_per=l_offspring_per.at[ind].set(sum_nb_offspring/nb_death)
    l_food_cons_per=l_food_cons_per.at[ind].set(sum_nb_food/nb_death)
    
    
    return l_expectancy,l_death,l_offspring,l_offspring_per,l_food_cons_per
update_life_fn=jax.jit(update_life)  

@dataclass
class AgentStates_noparam(object):
    posx: jnp.uint16
    posy: jnp.uint16
    energy: jnp.ndarray
    time_good_level: jnp.uint16
    time_alive: jnp.uint16
    time_under_level: jnp.uint16
    alive: jnp.int8


@dataclass
class State_noparam(TaskState):
    last_actions: jnp.int8
    rewards: jnp.int8
    agents: AgentStates_noparam
    steps: jnp.int32

def state_no_params(state):
    agents=state.agents
    new_state_no_param=State_noparam(last_actions=state.last_actions, rewards=state.rewards,
          agents=AgentStates_noparam(posx=agents.posx, posy=agents.posy, energy=agents.energy, time_good_level=agents.time_good_level,
                                     time_alive=agents.time_alive, time_under_level=agents.time_under_level, alive=agents.alive),
          steps=state.steps)
    return new_state_no_param

state_no_params_fn=jax.jit(state_no_params)

def place(i,a,l_movement,valid):
    a=a.at[i,l_movement[i]].add(valid[i])
    
    return a
place_fn=jax.jit(place)

def train(project_dir):
    with open(project_dir + "/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    max_steps = config["num_gens"] * config["gen_length"] + 1

    # initialize environment
    env = Gridworld(SX=config["grid_length"],
                    SY=config["grid_width"],
                    nb_agents=config["nb_agents"],
                    regrowth_scale=config["regrowth_scale"],
                    niches_scale=config["niches_scale"],
                    max_age=config["max_age"],
                    time_reproduce=config["time_reproduce"],
                    time_death=config["time_death"],
                    energy_decay=config["energy_decay"],
                    spontaneous_regrow=config["spontaneous_regrow"],
                    wall_kill=config["wall_kill"],
                    )
    key = jax.random.PRNGKey(config["seed"])
    next_key, key = random.split(key)
    state = env.reset(next_key)

    # initialize policy

    keep_mean_rewards = []
    keep_max_rewards = []
    #eval_params = []


    gens = list(range(config["num_gens"]))
    
    nb_gens=config["num_gens"]
    spx=jnp.zeros((8,100*nb_gens))
    spfx=jnp.zeros((8,100*nb_gens))
    lx=jnp.zeros((100*nb_gens))
    ly=jnp.zeros((100*nb_gens))
    lfx=jnp.zeros((100*nb_gens))
    rewards=jnp.zeros((100*nb_gens))
    lalive=jnp.zeros((100*nb_gens))
    lfood=jnp.zeros((100*nb_gens))

    l_movement=jnp.zeros((20*nb_gens,1000))
    l_valid=jnp.zeros((20*nb_gens,1000))

    l_expectancy=jnp.zeros((nb_gens*2))
    l_death=jnp.zeros((nb_gens*2))
    l_offspring=jnp.zeros((nb_gens*2))

    l_offspring_per=jnp.zeros((nb_gens*2))
    l_food_cons_per=jnp.zeros((nb_gens*2))
    sum_time_alive=0
    nb_death=0
    total_offspring=0

    sum_nb_offspring=0
    sum_nb_food=0

    prev_state=state
    last_alive=state.agents.alive


    for gen in gens:
        if(state.agents.alive.sum()==0):
            print("All the population died")
            break

        mean_accumulated_rewards=0

        if gen % config["eval_freq"] == 0:
            vid = VideoWriter(project_dir + "/train/media/gen_" + str(gen) + ".mp4", 20.0)
            state_log = []
            #state_grid_log = []
        # start = time.time()
        for i in range(config["gen_length"]):

                state, reward = env.step(state)
                sum_time_alive,nb_death,total_offspring,sum_nb_food,sum_nb_offspring=compute_death_offspring(state.agents.time_alive,state.agents.alive,last_alive,state.agents.nb_food,state.agents.nb_offspring,sum_time_alive,nb_death,total_offspring,sum_nb_food,sum_nb_offspring)
                last_alive=state.agents.alive*1.0
                
              
                if(i%10==0):
                  lx,ly,lfx,rewards,lalive,spx,spfx,lfood=update_log_fn(gen*100+i//10,state,lx,ly,lfx,rewards,lalive,spx,spfx,lfood)
                
                if(i%50==0):
                  l_movement,l_valid=update_movement_fn(gen*20+i//50,state,prev_state,l_movement,l_valid)
                  
                  #l_expectancy=update_expectancy_fn(j*20+i//50,state,prev_state,l_expectancy)
                  prev_state=state

                if(i%500==0):
                    #print(nb_death,nb_offspring)
                    l_expectancy,l_death,l_offspring,l_offspring_per,l_food_cons_per=update_life_fn(gen*2+i//500,sum_time_alive,nb_death,total_offspring,sum_nb_food,sum_nb_offspring,l_expectancy,l_death,l_offspring,l_offspring_per,l_food_cons_per)
                    sum_time_alive=0
                    nb_death=0
                    total_offspring=0
                    sum_nb_food=0
                    sum_nb_offspring=0
               


                    

                if (gen % config["eval_freq"] == 0):
                    rgb_im = state.state[:, :, :3]

                    rgb_im=jnp.clip(rgb_im,0,1)
                  
                
                     #white green and black
                    rgb_im=jnp.clip(rgb_im+jnp.expand_dims(state.state[:,:,1],axis=-1),0,1)
                    rgb_im=rgb_im.at[:,:,1].set(0)
                    rgb_im= 1-rgb_im
                    rgb_im=rgb_im-jnp.expand_dims(state.state[:,:,0],axis=-1)
                    
                    rgb_im = np.repeat(rgb_im, 2, axis=0)
                    rgb_im = np.repeat(rgb_im, 2, axis=1)
                    
                    vid.add(rgb_im)

                    mean_accumulated_rewards = mean_accumulated_rewards + (reward*state.agents.alive).sum()/(state.agents.alive.sum()+1e-10)

                    state_log.append(state_no_params_fn(state))
                    #state_grid_log.append(jnp.bool_(state.state[:,:,1]))

        # print("Training ", str(config["gen_length"]), " steps took ", str(time.time() - start))

                    



        if gen % config["eval_freq"] == 0:

            keep_mean_rewards.append(mean_accumulated_rewards)


            vid.close()
            #with open(project_dir + "/train/data/gen_" + str(gen) + "states_grid.npy", "wb") as f:
            #    jnp.save(f,jnp.bool_(jnp.array(state_grid_log)))
            with open(project_dir + "/train/data/gen_" + str(gen) + "states.pkl", "wb") as f:
                pickle.dump({"states": state_log}, f)
            save_model(model_dir=project_dir + "/train/models", model_name="gen_" + str(gen), params=state.agents.params)

            plt.plot(range(len(keep_mean_rewards)), keep_mean_rewards, label="mean")
            plt.ylabel("Mean Training rewards")
            plt.legend()
            plt.savefig(project_dir + "/train/media/rewards_" + str(gen) + ".png")
            plt.clf()



    n=nb_gens*20
    a=jnp.zeros((n,51))
    l_movement=l_movement.astype(jnp.int32)

    valid=(l_valid>0).astype(jnp.int32)
    for i in range(n):
        a=place_fn(i,a,l_movement,valid)
    for c in range(20):
        plt.plot(400-lx[c*5000:(c+1)*5000],label="avg x position")
        plt.plot(400-lfx[c*5000:(c+1)*5000],label="avg x food")
        plt.ylim(0,400)
        plt.legend()
        plt.savefig(project_dir + "/train/media/pos_" + str(c) + ".png")
        plt.clf()

        plt.plot(rewards[c*5000:(c+1)*5000])
        plt.savefig(project_dir + "/train/media/rewards_true_" + str(c) + ".png")
        plt.clf()

        plt.plot(lalive[c*5000:(c+1)*5000])
        plt.savefig(project_dir + "/train/media/alive_" + str(c) + ".png")
        plt.clf()




        plt.imshow(spx[:,c*5000:(c+1)*5000]*4, interpolation='nearest', aspect='auto')
        plt.savefig(project_dir + "/train/media/spx_" + str(c) + ".png")
        plt.clf()



        plt.imshow(spfx[:,c*5000:(c+1)*5000]*4, interpolation='nearest', aspect='auto')
        plt.savefig(project_dir + "/train/media/spfx_" + str(c) + ".png")
        plt.clf()

        plt.plot(lfood[c*5000:(c+1)*5000])
        plt.savefig(project_dir + "/train/media/food_" + str(c) + ".png")
        plt.clf()

        plt.plot(l_expectancy[c*100:(c+1)*100])
        plt.savefig(project_dir + "/train/media/expectancy_" + str(c) + ".png")
        plt.clf()

        plt.plot(l_death[c*100:(c+1)*100])
        plt.savefig(project_dir + "/train/media/death_" + str(c) + ".png")
        plt.clf()

        plt.plot(l_offspring[c*100:(c+1)*100])
        plt.savefig(project_dir + "/train/media/offspring_" + str(c) + ".png")
        plt.clf()

        plt.plot(l_offspring[c*100:(c+1)*100]/(lalive[::50][c*100:(c+1)*100]+1e-10))
        plt.savefig(project_dir + "/train/media/offspring_per_indiv" + str(c) + ".png")
        plt.clf()

        plt.scatter(jnp.ravel(l_valid[c*1000:(c+1)*1000])[l_valid[c*1000:(c+1)*1000].ravel()>0],jnp.ravel(l_movement[c*1000:(c+1)*1000])[l_valid[c*1000:(c+1)*1000].ravel()>0],s=0.001)
        plt.savefig(project_dir + "/train/media/movement_scatter_" + str(c) + ".png")
        plt.clf()


        plt.imshow(a[c*1000:(c+1)*1000].T, interpolation='nearest', aspect='auto')
        plt.savefig(project_dir + "/train/media/movement_" + str(c) + ".png")
        plt.clf()

        plt.imshow(jnp.log(a[c*1000:(c+1)*1000]+1).T, interpolation='nearest', aspect='auto')
        plt.savefig(project_dir + "/train/media/movement_log_" + str(c) + ".png")
        plt.clf()
        
        
        plt.plot(l_offspring_per[c*100:(c+1)*100])
        plt.savefig(project_dir + "/train/media/nb_offspring_per_indiv_" + str(c) + ".png")
        plt.clf()

        plt.plot(l_food_cons_per[c*100:(c+1)*100])
        plt.savefig(project_dir + "/train/media/nb_food_per_indiv" + str(c) + ".png")
        plt.clf()


            

    plt.plot(400-lx,label="avg x position")
    plt.plot(400-lfx,label="avg x food")
    plt.ylim(0,400)
    plt.legend()
    plt.savefig(project_dir + "/train/media/pos_" + ".png")
    plt.clf()

    plt.plot(rewards)
    plt.savefig(project_dir + "/train/media/rewards_true" + ".png")
    plt.clf()

    plt.plot(lalive)
    plt.savefig(project_dir + "/train/media/alive_"  + ".png")
    plt.clf()


    plt.plot(lfood)
    plt.savefig(project_dir + "/train/media/food_"  + ".png")
    plt.clf()
    
    plt.plot(l_expectancy)
    plt.savefig(project_dir + "/train/media/expectancy_"  + ".png")
    plt.clf()

    plt.plot(l_death)
    plt.savefig(project_dir + "/train/media/death_"  + ".png")
    plt.clf()

    plt.plot(l_offspring)
    plt.savefig(project_dir + "/train/media/offspring_"  + ".png")
    plt.clf()

    plt.plot(l_offspring/(lalive[::50]+1e-10))
    plt.savefig(project_dir + "/train/media/offspring_per_indiv"  + ".png")
    plt.clf()



    plt.scatter(jnp.ravel(l_valid)[l_valid.ravel()>0],jnp.ravel(l_movement)[l_valid.ravel()>0],s=0.001)
    plt.savefig(project_dir + "/train/media/movement_scatter"  + ".png")
    plt.clf()

    plt.imshow(a.T, interpolation='nearest', aspect='auto')
    plt.savefig(project_dir + "/train/media/movement_"+ ".png")
    plt.clf()

    plt.imshow(jnp.log(a+1).T, interpolation='nearest', aspect='auto')
    plt.savefig(project_dir + "/train/media/movemeng_log_"+ ".png")
    plt.clf()




    plt.imshow(spx*4, interpolation='nearest', aspect='auto')
    plt.savefig(project_dir + "/train/media/spx_"+ ".png")
    plt.clf()



    plt.imshow(spfx*4, interpolation='nearest', aspect='auto')
    plt.savefig(project_dir + "/train/media/spfx_" + ".png")
    plt.clf()
    
    plt.plot(l_offspring_per)
    plt.savefig(project_dir + "/train/media/nb_offspring_per_indiv" + ".png")
    plt.clf()

    plt.plot(l_food_cons_per)
    plt.savefig(project_dir + "/train/media/nb_food_per_indiv" + ".png")
    plt.clf()
    
    
    jnp.save(project_dir + "/train/data/alive",lalive)
    jnp.save(project_dir + "/train/data/food",lfood)
    jnp.save(project_dir + "/train/data/movement",l_movement)
    jnp.save(project_dir + "/train/data/valid",l_valid)
    jnp.save(project_dir + "/train/data/expectancy",l_expectancy)
    jnp.save(project_dir + "/train/data/death",l_death)
    jnp.save(project_dir + "/train/data/offspring",l_offspring)
    jnp.save(project_dir + "/train/data/offspring_per_indiv",l_offspring_per)
    jnp.save(project_dir + "/train/data/food_cons_per_indiv",l_food_cons_per)
    jnp.save(project_dir + "/train/data/spx",spx)
    jnp.save(project_dir + "/train/data/spfx",spfx)
    jnp.save(project_dir + "/train/data/lx",lx)
    jnp.save(project_dir + "/train/data/lfx",lfx)
    

            # run offline evaluation
            #eval_params.append(eval(state.agents.params, config["nb_agents"], key, env.model, project_dir, config["agent_view"], gen))
            #process_eval(eval_params, project_dir, gen)

    



if __name__ == "__main__":
    project_dir = sys.argv[1]
    if not os.path.exists(project_dir + "/train/data"):
        os.makedirs(project_dir + "/train/data")

    if not os.path.exists(project_dir + "/train/models"):
        os.makedirs(project_dir + "/train/models")

    if not os.path.exists(project_dir + "/train/media"):
        os.makedirs(project_dir + "/train/media")
    train(project_dir)
