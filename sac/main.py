import sac.utils
from sac.utils import timing, profile
import numpy as np
import scipy.sparse as sps
import pdb
import os
from matplotlib import pyplot as plt
from line_profiler import LineProfiler
import sac.equations
equations = sac.equations.get_equations()
bp = pdb.set_trace


# %% FUNCTIONS
class Network(object):
    @profile
    def __init__(self, trials, stimInfo, stimColumns, **kwargs):
        trials.onset = trials.onset + 0.001
        trials['trial'] = np.arange(trials.shape[0])+1
        self.trials = trials
        self.onsets = trials.onset.tolist()
        self.stimColumns = stimColumns
        self.create_stim_array()
        self.subj = self.trials.subject.unique()
        self.onsets = np.array(self.trials.onset)    
        self.build_time_matrix()
        self.build_association_matrix()
        self.build_event_matrixes()
        self.extract_prior_base(stimInfo)     
        self.tSinceLastEvent = {}
        self.get_time_since_last_event('concepts')
        self.get_time_since_last_event('episodes')
        self.create_nodes_and_links()
        self.build_node_look_up_table()
        self.create_wm_module()
        self.prepare_results_df()
        
    @profile   
    def run_trials(self, par, equations, savePars=False, **kwargs):
        self.check_pars(par)
        self.equations = equations
        self.reset_nodes_and_links()

        for i in range(len(self.trials.onset)):
            # speed up by zipping over things in update_time?
            self.update_time(i)
            self.update_base()
            self.update_activation(nodeType="concepts")
            self.update_wm()
            self.update_activation(nodeType="episodes")
            self.activate_perceived_nodes()
            self.spread_activation()
            self.concepts.strengthen_base(self, self.cEvent, equations['conceptLearning'], self.par['p'])
            self.episodes.strengthen_base(self, self.cEpiEvent, equations['episodeLearning'], self.par['p'])
            self.links.strengthen_base(self, self.cLinkEvent, equations['linkLearning'], self.par['p'])

        self.save_node_values_to_trialdf()
        if savePars: self.save_paramters_to_trialdf()
        
    def check_pars(self, par):
        """ 
        if a non-essential parameter is not specified, give it a default.
        Also check for incorrect parameters
        """
        self.par = par
        if self.par.get('dn',0) > 0 or self.par.get('dl',0) >= 0:
            raise ValueError('Decay rate parameter should be negative!')
        if self.par.get('dn_sem',0) > 0:
            raise ValueError('Decay rate parameter should be negative!')
        if self.par.get('dn_epi',0) > 0:
            raise ValueError('Decay rate parameter should be negative!')            
            
        # defaults. @TODO: add full defaults
        if 'p_forget_prop' not in self.par.keys():
            self.par['p_forget_prop'] = 0.1
            
        # if there is a noise recovery parameter, draw recovery from a distribution
        if self.par.get('recovery_noise', False):
            self.par['w_recovery_rate'] = np.random.gamma(10*self.par['w_recovery_rate']/self.par['recovery_noise'], self.par['recovery_noise'])/10
            
            
    def extract_prior_base(self, stimInfo):
        """ keeps only the words that are shown to the subject """
        idx = self.T.index.values
        stimInfo = stimInfo.loc[idx]
        self.cPriorB = np.array(stimInfo.priorBase).T
        self.cPriorFan = np.array(stimInfo.priorFan).T 
        self.contextIdx = stimInfo.index.str.contains('context')
        self.cPriorB[self.contextIdx] = [1] * len(self.cPriorB[self.contextIdx])
                 
    def create_stim_array(self):
        """ saves columns with stim in a separate array to use later in save_node_values_to_trialdf() """
        cols = [col for col in self.stimColumns if col != 'list']
        self.stimArray = self.trials[cols].values

    def build_time_matrix(self):
        """ Create a time matrix to hold when events occured """
#        import pdb; pdb.set_trace()
        T = self.trials.filter(regex='stim|trial|onset|list')
        T1 = T.melt(id_vars=['trial', 'onset'])
        T1.onset = T1.onset
        T1 = T1.filter(regex='trial|onset|value')
        T1 = T1.drop_duplicates()
        self.T = T1.pivot(index='value', columns='trial', values='onset')
        
        # remove timestamps when one cell is empty (e.g. nan in column)
        self.TplusNan = self.T.copy()
        if sum(self.T.index == 'nan') > 0:
            self.T.loc['nan'] = np.nan
        self.Tsparse = sac.utils.array_to_sparse(self.T.values)
        self.TsparseNeg = -self.Tsparse
        uniqueEpisodes = T.copy()
        uniqueEpisodes['concat'] = ''
        self.trials['epiname'] = ''
        for col in self.stimColumns:
            uniqueEpisodes['concat'] = uniqueEpisodes['concat'] + uniqueEpisodes[col].astype(str) + '.'
            self.trials['epiname'] = self.trials['epiname'] + self.trials[col].astype(str) + '.'
        self.episodeLookUpTable = uniqueEpisodes[self.stimColumns+['concat']]
        uniqueEpisodesSubset = uniqueEpisodes[['onset','trial','concat']]
        self.ET = uniqueEpisodesSubset.pivot(index='concat', columns='trial', values='onset')
        self.ETsparseNeg = -sac.utils.array_to_sparse(self.ET.values)
        self.trialIds = range(0,self.trials.shape[0])
        
        self.ones = [np.ones(n) for n in self.trialIds]
        self.listTsparseNeg = [-sac.utils.array_to_sparse(self.T.values[:,0:trial].T).tocsr() for trial in self.trialIds]
        self.listETsparseNeg = [-sac.utils.array_to_sparse(self.ET.values[:,0:trial].T).tocsr() for trial in self.trialIds]        
        
        
    def build_node_look_up_table(self):
        # dict in which each key is a semantic node, and value is a list of 
        # episode nodes to which it is connected
        d = {}
        self.episodeLookUpTable = (self.episodeLookUpTable.
            replace('nan',np.nan).
            dropna().
            drop_duplicates())
        for i, row in self.episodeLookUpTable.iterrows():
            for col in self.stimColumns:
                item = row[col]
                d[item] = d.get(item,[]) + [row['concat']]
        self.episodeLookUpTable = d
        self.episodeIdxDict = {key: i for i, key in enumerate(self.ET.index.values)}

    def build_association_matrix(self):
        """  Makes an array that shows when an item and an episode have coocurred. """
        nItems = self.T.shape[0]
        nEvents = self.T.shape[1]
        nEpisodes = self.ET.shape[0]
        events3d = np.empty([nEvents,nEpisodes,nItems])
        shape3d = events3d.shape
        outmat = np.empty([nEpisodes,nItems])
#        bp()
        for i in range(self.T.shape[1]):
            events3d[i, :, :] = np.outer(self.ET.iloc[:, i], self.T.iloc[:, i], out=outmat)
        events3d = events3d.reshape(shape3d[0], shape3d[1]*shape3d[2])
        events3d = sac.utils.array_to_sparse(events3d)
        events3d = events3d.T
        events3d = events3d.power(0.5)
        self.shape3d = shape3d
        self.linkTneg = -events3d
        transposed = self.linkTneg.T.tocsr()
        self.listLinkTneg = [transposed[0:trial,:] for trial in self.trialIds]
        self.nConcepts = nItems-1
        self.nEvents = nEvents
        self.nEpisodes = nEpisodes
        
        
    def build_event_matrixes(self):
        """
        2d array with stim in rows and timepoints in columns, showin 1 if the
        stim occured and 0 otherwise
        """
        events = (1-np.isnan(self.T.values))
        self.eventsArray = events
        self.eventsSparse = sac.utils.array_to_sparse(events)
        self.epiEventsArray = 1-np.isnan(self.ET.values)
        self.epiEventsSparse = sac.utils.array_to_sparse(1-np.isnan(self.ET.values))
        self.linkEvents = (self.linkTneg < 0)
        self.linkEvents = self.linkEvents.asformat('csc')
        self.listLinkEvents = [self.linkEvents[:,i] for i in self.trialIds]


    def create_nodes_and_links(self):
        self.concepts = Nodes(self.Tsparse, self.cPriorB)
        self.episodes = Nodes(self.ETsparseNeg)
        self.links = Links(self)

    def create_wm_module(self):
        self.wmAvailable = np.zeros((self.trials.shape[0],))
        self.wmSpent = np.zeros((self.trials.shape[0]),)
        
    def prepare_results_df(self):
        self.results = self.trials.copy()
        for col in self.stimColumns:
            self.results[col+'.semB'] = np.nan
            self.results[col+'.semA'] = np.nan
        self.results['epiA'] = np.nan
        self.results['epiB'] = np.nan
        self.results['retrievedName'] = ''
        self.results['sameNode'] = False
        self.results['relatedNode'] = False
        self.results['wmSpent'] = 0.   
        self.results = self.results.to_dict('list')

    def reshape_link_B_to_2d_array(self):
        return(self.links.B.reshape((self.shape3d[1], self.shape3d[2])))

    def normalize_link_strength_by_fan(self):
        linksB = self.reshape_link_B_to_2d_array()
#        x = sac.utils.array_to_sparse(linksB).tolil().T
#        y = x.sum(axis=0)
#        y = y+self.cPriorFan
#        y[y==0]=1
#        x = x/y
        linksBPlusPrior = linksB.sum(axis=0)
        linksBPlusPrior += self.cPriorFan
        linksBPlusPrior[linksBPlusPrior == 0] = 1
        linksB = (linksB / linksBPlusPrior).T
        return(linksB)

    def reset_nodes_and_links(self):
        self.concepts.S = self.concepts.S.asformat('csr')
        self.concepts.S *= 0
        self.concepts.S.eliminate_zeros()
        self.concepts.S = self.concepts.S.asformat('lil')
        self.concepts.B *= 0
        self.concepts.A *= 0
        self.episodes.B *= 0
        self.episodes.A *= 0
        self.episodes.S = self.episodes.S.asformat('csr')
        self.episodes.S *= 0
        self.episodes.S.eliminate_zeros()
        self.episodes.S = self.episodes.S.asformat('lil')
        self.links.S = self.links.S.asformat('csr')
        self.links.S *= 0
        self.links.S.eliminate_zeros()
        self.links.S = self.links.S.asformat('lil')
        self.links.B *= 0
        self.concepts.lastA = np.array(self.concepts.Bprior)
        self.episodes.lastA = self.episodes.lastA*0
        self.wmAvailable = [0] * len(self.wmAvailable)
        self.wmSpent = [0] * len(self.wmSpent)
        self.prepare_results_df()

    def update_time(self, trial):
        """ updates the internal time of the network """
        self.cTrial = trial
        self.cTime = self.onsets[trial]
        if 'duration' in self.trials.columns:
            self.cDuration = self.trials['duration'][trial]
        self.cEvent = self.eventsArray[:, trial]
        self.cEpiEvent = self.epiEventsArray[:, trial]
        self.cLinkEvent = self.listLinkEvents[trial]

    def get_time_since_last_event(self, nodeType):
        """ for each concept, return how long ago it was experienced lastly """
        if nodeType == "concepts":
            timeDF = self.T
        else:
            timeDF = self.ET
        tArray = np.array(timeDF)
        tArray[np.isnan(tArray)] = 0
        for col in range(1, tArray.shape[1]):
            tArray[:, col] = np.amax(tArray[:,0:(col+1)], axis=1)
        tArray[tArray==0] = np.nan
        tmp = np.array([[np.nan] * tArray.shape[0]]).T
        tArray = np.hstack((tmp, tArray))
        tArray = tArray[:,range(tArray.shape[1]-1)]
        onsets = np.outer(np.ones(timeDF.shape[0]), self.onsets)
        self.tSinceLastEvent[nodeType] = tArray - onsets
        self.tSinceLastEvent[nodeType][np.isnan(self.tSinceLastEvent[nodeType])] = 0

    def request_wm(self):
        """ requests resources for activation and returns what is available """
#        if self.cTrial >= 10: bp()
        currentAct = self.concepts.A[:, self.cTrial]
        remainingAct = self.par['sem_theta'] - currentAct
        wmRequested = self.cEvent * remainingAct ** self.par['w_act']
        wmReturnedTotal = min(wmRequested.sum(), self.wmAvailable[self.cTrial])
        wmPerConcept = wmRequested * wmReturnedTotal / wmRequested.sum()
        self.wmSpent[self.cTrial] = wmReturnedTotal
#        self.wmAvailable[self.cTrial] = self.wmAvailable[self.cTrial]-self.wmSpent[self.cTrial]
        return(wmPerConcept)

    def update_wm(self):
        """ calculates how much resources are available at trial start """
        if self.cTrial != 0:
            tSinceLastTrial = self.onsets[self.cTrial] - self.onsets[self.cTrial-1]
            recovery_rate = self.par['w_recovery_rate']    
            wmRestored = tSinceLastTrial * recovery_rate
            wmNew = (self.wmAvailable[self.cTrial-1]
                     - self.wmSpent[self.cTrial-1]
                     + wmRestored)
            self.wmAvailable[self.cTrial] = min(wmNew, self.par['W'])
        else:
            self.wmAvailable[self.cTrial] = self.par['W']

    def update_base(self):
        """ updates the base-level strength of all nodes. No strengthening """
        cT = self.listTsparseNeg[self.cTrial]
        eT = self.listETsparseNeg[self.cTrial]
        lT = self.listLinkTneg[self.cTrial]
        ones = self.ones[self.cTrial]
        self.concepts.B[:, self.cTrial] = self.eval_base_sparse(self.concepts.S,
                                                           cT,
                                                           self.cTime,
                                                           self.par['dn'],
                                                           self.concepts.Bprior,
                                                           self.equations['conceptDecay'],
                                                           ones)
        self.episodes.B[:, self.cTrial] = self.eval_base_sparse(self.episodes.S,
                                                           eT,
                                                           self.cTime,
                                                           self.par['dn'],
                                                           0,
                                                           self.equations['episodeDecay'],
                                                           ones)
        self.links.B = self.eval_base_sparse(self.links.S,
                                        lT,
                                        self.cTime,
                                        self.par['dl'],
                                        0,
                                        self.equations['linkDecay'],
                                        ones)

        
        
    def eval_base_sparse(self, S, T, cTime, d, priorB, equation, ones, structure='nodes'):
        """
        Calculates the current base activation of all nodes/links, based on the
        time of occurences (T), the increments at each time (S), and the current
        time (cTime).
    
        Parameters
        -----------
        T : a matrix where each row is a single sitmulus vector, and each column
            a single trial vector. Cells contain the timestamp of when the event
            occured (if only one stimulus is displayed on a trial, it is a column
            with 0s for all other stimuli, and the timing for the presented stim)
        S : similar to T in shape and size, but stores the size of the increment at
            each occurence
        cTime: a scalar for the current time
        priorB: prexisting base activation. Default is zero, but you can supply
                a pandas series with the calculated prior base for concept nodes
        structure: string
            'links' or 'nodes'
        """
        T1=T.copy()
        B = equations[equation](S=S, T=T1, t=cTime, d=d, priorB=priorB, ones=ones)
        return(B)


#    def update_activation(self, nodeType):
#        """ calculates the current activation value of nodes """
#        bp()
#        if (nodeType == "concepts"):
#            nodes = self.concepts
#        else:
#            nodes = self.episodes
#        tSinceLastEvent = self.tSinceLastEvent[nodeType][:, self.cTrial]
#        decayMultiplier = np.exp(self.par['y'] * tSinceLastEvent)
#        if self.par.get('multiply_activation',False) & (nodeType == "episodes"):
#            deltaAct = nodes.lastA
#        else:
#            deltaAct = (nodes.lastA
#                        - nodes.B[:, self.cTrial])
#        decayedDeltaAct = deltaAct * decayMultiplier
#        nodes.A[:, self.cTrial] = nodes.B[:, self.cTrial] + decayedDeltaAct
#        if (nodeType == "concepts"):
#            nodes.A[self.contextIdx, self.cTrial] = self.par['contextAct']
        
    def update_activation(self, nodeType):
        """ calculates the current activation value of nodes """
        if (nodeType == "concepts"):
            nodes = self.concepts
        else:
            nodes = self.episodes
        decayMultiplier = self.par['y'] ** max((self.onsets[self.cTrial]-self.onsets[self.cTrial-1]),0)
        if self.par.get('multiply_activation',False) & (nodeType == "episodes"):
            deltaAct = nodes.A[:, self.cTrial-1]
        else:
            if self.cTrial != 0:
                deltaAct = (nodes.A[:, self.cTrial-1] - nodes.B[:, self.cTrial])
            else:
                deltaAct = nodes.B[:, self.cTrial]-nodes.B[:, self.cTrial]
        decayedDeltaAct = deltaAct * decayMultiplier
        nodes.A[:, self.cTrial] = decayedDeltaAct
        if (nodeType == "concepts"):
            nodes.A[:, self.cTrial] = nodes.B[:, self.cTrial] + decayedDeltaAct
            nodes.A[self.contextIdx, self.cTrial] = self.par['contextAct']
            


    def activate_perceived_nodes(self):
        """ activate the perceived nodes up to threshold """
        wmReceived = self.request_wm()
        self.concepts.initA = self.concepts.A[:, self.cTrial].copy()
        self.concepts.A[:, self.cTrial] += (wmReceived ** (1/self.par['w_act']))
        idx = self.cEvent == 1
        self.concepts.lastA[idx] = self.concepts.A[idx, self.cTrial]
        self.episodes.lastA[self.cEpiEvent == 1] = self.episodes.A[self.cEpiEvent == 1, self.cTrial]

    def spread_activation(self):
        """ spreads activation to connected nodes """
        normedLinkB = self.normalize_link_strength_by_fan()
        activation = self.cEvent * self.concepts.A[:, self.cTrial]
        # if the test is free recall, don't spread activation from sem node
        if 'testtype' in self.trials.columns:
            if self.trials['testtype'][self.cTrial] == 'free_recall':
                activation = activation * self.contextIdx
        spreadingAct = activation.dot(normedLinkB)
        if self.par.get('multiply_activation',False):
            self.episodes.A[:, self.cTrial] = self.episodes.A[:, self.cTrial] + self.episodes.B[:, self.cTrial] * (spreadingAct)
        else:
#            self.episodes.A[:, self.cTrial] = self.episodes.B[:, self.cTrial] + spreadingAct
            self.episodes.A[:, self.cTrial] = self.episodes.A[:, self.cTrial] + spreadingAct

        # don't spread activation to nodes that do not exist, i.e. base=0
        existingNodes = self.episodes.B[:, self.cTrial] > 0.000001
        if not self.par.get('spread_to_zero', False):
            self.episodes.A[:, self.cTrial] *= existingNodes

    def extract_epi_activation(self):
        """ extracts the episodic node activation at the begining of trial """
        # prepare structures to save the data
        episodeB = self.episodes.B
        episodeA = self.episodes.A
        episodeS = self.episodes.S
        epiB = [0] * self.trials.shape[0]
        epiA = [0] * self.trials.shape[0]
        epiS = [0] * self.trials.shape[0]
        retrievedName = [''] * self.trials.shape[0]
        nodeRelated = [False] * self.trials.shape[0]
        
        for i, stim in enumerate(self.stimArray):
            # extract the name of a node connected to at least one of the cues
            targetNodes = [self.episodeLookUpTable.get(cue,[]) for cue in stim]
            targetNodes = list(set(sum(targetNodes,[])))
            if self.results['epiname'][i] in targetNodes:
                idx = self.episodeIdxDict[self.results['epiname'][i]]
                epiA[i] = episodeA[idx,i]
                epiB[i] = episodeB[idx,i]
                epiS[i] = episodeS[i,idx]
                retrievedName[i] = self.results['epiname'][i]
                if epiA[i] > 0: 
                    continue
            # get activation of target nodes
            nTarget = len(targetNodes)
            if nTarget == 1:
                idx = self.episodeIdxDict[targetNodes[0]]
                epiA[i] = episodeA[idx,i]
                epiB[i] = episodeB[idx,i]
                epiS[i] = episodeS[i,idx]
                retrievedName[i] = targetNodes[0]
                nodeRelated[i] = True
            elif nTarget >= 1:
                epiAct = [0] * nTarget
                epiBase = [0] * nTarget   
                epiStr = [0] * nTarget    
                for k, node in enumerate(targetNodes):
                    idx  =  self.episodeIdxDict[node]
                    epiAct[k] = episodeA[idx,i]
                    epiBase[k] = episodeB[idx,i]
                    epiStr[k] = episodeS[i,idx]
                retrievedName[i] = targetNodes[np.array(epiAct).argmax()]
                epiA[i] = np.max(epiAct)
                epiB[i] = np.max(epiBase)
                epiS[i] = np.max(epiStr)
                nodeRelated[i] = True
        return({'epiA': epiA, 'epiB': epiB, 'epiS': epiS, 
                'retrievedName': retrievedName, 
                'nodeRelated': nodeRelated})

    def save_node_values_to_trialdf(self):
        # find concept strength and activation for each trial and save to trials
        e = self.TplusNan.values > 0
        e = e.flatten()
        B = self.concepts.B.flatten() 
        B = B[e]
        A = self.concepts.A.flatten()
        A = A[e]
        items = self.TplusNan.index.values
        items = np.repeat(items, (len(e)/len(items)))
        items = items[e]
        trial = self.TplusNan.columns.values
        trial = np.tile(trial, int(len(e)/len(trial)))
        trial = trial[e]
        for col in self.stimColumns:
            s = self.results[col]
            idx1 = np.in1d(items,s)
            idx1 = np.nonzero(idx1)[0]
            trials = trial[idx1]
            idx2 = np.argsort(trials)
            s = np.array(s)[trials[idx2]-1]
            idx3 = np.nonzero(s == items[idx1][idx2])[0]
            B1 = B[idx1][idx2][idx3]
            A1 = A[idx1][idx2][idx3]
            self.results[col+'.semB'] = B1
            self.results[col+'.semA'] = A1            
            
        # extract episode node base and activation at begining and end of trial
        start_act = self.extract_epi_activation()
                
        self.results['epiA'] = start_act['epiA']
        self.results['epiB'] = start_act['epiB']
        self.results['epiS'] = start_act['epiS']
        self.results['retrievedName'] = start_act['retrievedName']
        self.results['sameNode'] = [x==y for x,y in zip(self.results['retrievedName'], 
                                    self.results['epiname'])]
        self.results['relatedNode'] = start_act['nodeRelated']
        self.results['wmSpent'] = self.wmSpent
        self.results['wmAvailable'] = self.wmAvailable
        self.results = self.extract_results_from_net()
        
    def extract_results_from_net(self, **kwargs):
        import pandas as pd
        res = pd.DataFrame(self.results)
        old_cols = self.trials.columns.values.tolist()
        new_cols = res.columns.values
        new_cols = new_cols[~np.in1d(new_cols, old_cols)].tolist()
        res = res[old_cols+new_cols]
        return(res)    

    def save_paramters_to_trialdf(self):
        for key, value in self.par.items():
            self.results[key] = value
            
    def save_matrices(self, directory):
        if not os.path.exists(directory):
            os.mkdir(directory)
        mats = [self.concepts.A, self.concepts.B, self.concepts.S,
                self.episodes.A, self.episodes.B, self.episodes.S,
                self.reshape_link_B_to_2d_array(), self.links.S]
        filenames = ['conceptA.csv', 'conceptB.csv', 'conceptS.csv',
                     'episodeA.csv', 'episodeB.csv', 'episodeS.csv',
                     'linksB.csv', 'linksS.csv']
        filenames = [directory + '\\' + filename for filename in filenames]
        for mat, file in zip(mats, filenames):
            if not type(mat) is np.ndarray:
                mat = mat.toarray()
            mat = np.round(mat, 3)
            np.savetxt(file, mat, delimiter=',')

    def plot_wm(self, start, end):
        idx = slice(start, end)
        plt.plot(self.wmAvailable[idx])
        
    def plot_concept_act(self, conceptid):
        plt.plot(self.concepts.A[conceptid,:].tolist())
    


class Nodes(object):
    def __init__(self, T, Bprior=None):
        # create matrices for storing node properties
        self.S = sps.lil_matrix(T*0)
        self.B = self.S.toarray().copy()
        self.A = self.B.copy()
        self.S = self.S.T
        if (Bprior is None):
            Bprior = np.array([0] * T.shape[0])
        self.Bprior = Bprior
        self.lastA = np.array(Bprior.copy())
        self.initA = np.array(Bprior.copy())

    def strengthen_base(self, net, event, equation, learning_rate):
        """ strengthens the base of experienced nodes """
        idx = np.nonzero(event)[0]
        inc = equations[equation](learningRate=learning_rate,
                                        baseLevel=self.B[idx, net.cTrial],
                                        net=net)
        self.S.data[net.cTrial] = self.S.data[net.cTrial] + inc.tolist()
        self.S.rows[net.cTrial] = self.S.rows[net.cTrial] + idx.tolist()

class Links(object):
    def __init__(self, network):
        self.S = sps.lil_matrix(network.linkTneg*0)
        self.B = self.S[:, 0].toarray()*0
        self.S = self.S.T

    def strengthen_base(self, net, event, equation, learning_rate):
        """ strengthens the experienced links """
        idx = event.indices
        inc = equations[equation](learningRate=learning_rate,
                                        baseLevel=self.B[idx],
                                        net=net)
        self.S.data[net.cTrial] = self.S.data[net.cTrial] + inc.tolist()
        self.S.rows[net.cTrial] = self.S.rows[net.cTrial] + idx.tolist()


if __name__ == '__main__':  
    path = 'D:\\gdrive\\research\\projects\\122-sac-modelling\\'
        
    """ DEFINE CONSTANTS TO USE IN THE MODEL """
    initPars = {'SCALE': 1000.0,
       'W': 4,
       'contextAct': 1,
       'decaytype': 'power',
       'dl': -0.12,
       'dn': -0.18,
       'fan': 2,
       'p': 0.65,
       'p_forget_prop': 0.1,
       'w_act': 1,
       'prior0': 0.4,
       'prior1': 0.2,
       'prior2': -0.1,
       'sem_theta': 1,
       'w_recovery_rate': 1,
       'y': 0.2,}
    
    """ SELECT WHICH EQUATIONS TO USE IN THE MODEL FITTING """
    eqs = {'conceptLearning': 'learning_equation4',
          'episodeLearning': 'learning_equation8',
          'linkLearning': 'learning_equation2',
          'conceptDecay': 'decay_power',
          'episodeDecay': 'decay_power',
          'linkDecay': 'decay_power'}
    
    filterCond = 'procedure == "test" and MemCuePostprioritem1 != "nan"'
    
    """ LOAD DATA FOR ALL SUBJECTS AND THE WHICH SUBJECTS TO FIT """
    allsubjects = sac.utils.load_trials(os.path.join(path, 'data/marevic2017_exp1.csv'), scale=initPars['SCALE'])
    trials = sac.utils.select_subjects(allsubjects, [1])
    stim_columns = ['stim1','stim2','list']  
    stimInfo = sac.utils.get_stim_info(allsubjects, stim_columns, initPars)
    
    net = Network(trials, stimInfo, stim_columns, duration=True)
    net.run_trials(initPars, eqs, duration=True)
#    %timeit net.run_trials(initPars, eqs, duration=True)
