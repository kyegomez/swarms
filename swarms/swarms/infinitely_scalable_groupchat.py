from swarms.embeddings.simple_ada import get_ada_embeddings
import chromadb
from swarms.models.openai_models import OpenAIChat

# Vectordb
client = chromadb.Client()
collection = client.create_collection(name="swarm")


def add_to_vectordb(task):
    """
    Add some text documents to the collection
    Chroma will store your text, and handle tokenization, embedding, and indexing automatically.

    """
    docs = collection.add(documents=[task], metadatas=[{"source": "agent1"}], ids=["1"])

    return docs


def query_vectordb(query: str):
    results = collection.query(query_texts=[query], n_results=1)
    return results


# Test
TASK_TEXT = """
11.3.1 Einstein’s A and B Coefficients
Picture a container of atoms,   of them in the lower state   , and   of them in the upper state   . Let A be the spontaneous emission rate,14 so that the number of particles leaving the upper state by this process, per unit time, is   .15 The transition rate for stimulated emission, as we have seen (Equation 11.54), is proportional to the energy density of the electromagnetic field:   , where ; the number of particles leaving the upper state by this mechanism, per unit time, is   . The absorption rate is likewise proportional to —call it   ; the number of particles per unit time joining the upper level is therefore . All told, then,
(11.55) Suppose these atoms are in thermal equilibrium with the ambient field, so that the number of particles in
each level is constant. In that case   , and it follows that
(11.56) On the other hand, we know from statistical mechanics16 that the number of particles with energy E, in
thermal equilibrium at temperature T, is proportional to the Boltzmann factor,   , so
(11.57)
       and hence
But Planck’s blackbody formula17 tells us the energy density of thermal radiation:
comparing the two expressions, we conclude that
and
(11.58)
(11.59)
(11.60)
(11.61)
    Equation 11.60 confirms what we already knew: the transition rate for stimulated emission is the same as for absorption. But it was an astonishing result in 1917—indeed, Einstein was forced to “invent” stimulated emission in order to reproduce Planck’s formula. Our present attention, however, focuses on Equation 11.61, for this tells us the spontaneous emission rate —which is what we are looking for—in terms of the stimulated emission rate   —which we already know. From Equation 11.54 we read off
(11.62)
     530
and it follows that the spontaneous emission rate is
(11.63)
Problem 11.10 As a mechanism for downward transitions, spontaneous emission competes with thermally stimulated emission (stimulated emission for which
  blackbody radiation is the source). Show that at room temperature ( thermal stimulation dominates for frequencies well below spontaneous emission dominates for frequencies well above mechanism dominates for visible light?
K) Hz, whereas Hz. Which
    Problem 11.11 You could derive the spontaneous emission rate (Equation 11.63) without the detour through Einstein’s A and B coefficients if you knew the ground state energy density of the electromagnetic field,   , for then it would simply be a case of stimulated emission (Equation 11.54). To do this honestly would require quantum electrodynamics, but if you are prepared to believe that the ground state consists of one photon in each classical mode, then the derivation is fairly simple:
  (a)
To obtain the classical modes, consider an empty cubical box, of side l, with one corner at the origin. Electromagnetic fields (in vacuum) satisfy the classical wave equation18
where f stands for any component of E or of B. Show that separation of variables, and the imposition of the boundary condition   on all six surfaces yields the standing wave patterns
with
There are two modes for each triplet of positive integers , corresponding to the two polarization states.
The energy of a photon is (Equation 4.92), so the energy in the mode   is
What, then, is the total energy per unit volume in the frequency range 531
    (b)
  
   (c)
What, then, is the total energy per unit volume in the frequency range , if each mode gets one photon? Express your answer in the form
and read off   . Hint: refer to Figure 5.3.
Use your result, together with Equation 11.54, to obtain the spontaneous
emission rate. Compare Equation 11.63.
  532

11.3.2 The Lifetime of an Excited State
Equation 11.63 is our fundamental result; it gives the transition rate for spontaneous emission. Suppose, now, that you have somehow pumped a large number of atoms into the excited state. As a result of spontaneous emission, this number will decrease as time goes on; specifically, in a time interval dt you will lose a fraction A dt of them:
  (assuming there is no mechanism to replenish the supply).19 Solving for   , we find:
evidently the number remaining in the excited state decreases exponentially, with a time constant
We call this the lifetime of the state—technically, it is the time it takes for   to reach initial value.
(11.64)
(11.65)
(11.66) of its
   I have assumed all along that there are only two states for the system, but this was just for notational simplicity—the spontaneous emission formula (Equation 11.63) gives the transition rate for
regardless of what other states may be accessible (see Problem 11.24). Typically, an excited atom has many different decay modes (that is:   can decay to a large number of different lower-energy states,   ,   ,   , ...). In that case the transition rates add, and the net lifetime is
(11.67)
Example 11.1
Suppose a charge q is attached to a spring and constrained to oscillate along the x axis. Say it starts out in the state (Equation 2.68), and decays by spontaneous emission to state   . From Equation 11.51 we have
You calculated the matrix elements of x back in Problem 3.39:
where ω is the natural frequency of the oscillator (I no longer need this letter for the frequency of the stimulating radiation). But we’re talking about emission, so   must be lower than n; for our purposes, then,
(11.68)
Evidently transitions occur only to states one step lower on the “ladder”, and the frequency of the 533
          
 Evidently transitions occur only to states one step lower on the “ladder”, and the frequency of the photon emitted is
(11.69) Not surprisingly, the system radiates at the classical oscillator frequency. The transition rate
 (Equation 11.63) is
and the lifetime of the nth stationary state is
Meanwhile, each radiated photon carries an energy   , so the power radiated is   :
(11.70)
(11.71)
(11.72)
     or, since the energy of an oscillator in the nth state is
,
 This is the average power radiated by a quantum oscillator with (initial) energy E.
For comparison, let’s determine the average power radiated by a classical oscillator with the same energy. According to classical electrodynamics, the power radiated by an accelerating charge q is given
by the Larmor formula:20
(11.73)
For a harmonic oscillator with amplitude   ,   , and the acceleration is . Averaging over a full cycle, then,
But the energy of the oscillator is   , so   , and hence
(11.74)
This is the average power radiated by a classical oscillator with energy E. In the classical limit
the classical and quantum formulas agree;21 however, the quantum formula (Equation 11.72) protects the ground state: If   the oscillator does not radiate.
      534

 Problem 11.12 The half-life   of an excited state is the time it would take for half the atoms in a large sample to make a transition. Find the relation between
and τ (the “lifetime” of the state).
∗ Problem 11.13 Calculate the lifetime (in seconds) for each of the four
states of hydrogen. Hint: You’ll need to evaluate matrix elements of the form ,   , and so on. Remember that   ,
, and   . Most of these integrals are zero, so inspect them closely before you start calculating. Answer:   seconds for all except   , which is infinite.
     535

    11.3.3 Selection Rules
The calculation of spontaneous emission rates has been reduced to a matter of evaluating matrix elements of the form
As you will have discovered if you worked Problem 11.13, (if you didn’t, go back right now and do so!) these quantities are very often zero, and it would be helpful to know in advance when this is going to happen, so we don’t waste a lot of time evaluating unnecessary integrals. Suppose we are interested in systems like hydrogen, for which the Hamiltonian is spherically symmetrical. In that case we can specify the states with the usual quantum numbers n,   , and m, and the matrix elements are
Now, r is a vector operator, and we can invoke the results of Chapter 6 to obtain the selection rules22 (11.75)
These conditions follow from symmetry alone. If they are not met, then the matrix element is zero, and the transition is said to be forbidden. Moreover, it follows from Equations 6.56–6.58 that
(11.76)
So it is never necessary to compute the matrix elements of both x and y; you can always get one from the other.
Evidently not all transitions to lower-energy states can proceed by electric dipole radiation; most are forbidden by the selection rules. The scheme of allowed transitions for the first four Bohr levels in hydrogen is shown in Figure 11.9. Notice that the   state   is “stuck”: it cannot decay, because there is no lower- energy state with . It is called a metastable state, and its lifetime is indeed much longer than that of, for example, the states ,   , and   . Metastable states do eventually decay, by collisions, or by “forbidden” transitions (Problem 11.31), or by multiphoton emission.
Figure 11.9: Allowed decays for the first four Bohr levels in hydrogen. 536
          
 Problem 11.14 From the commutators of   with x, y, and z (Equation 4.122): (11.77)
obtain the selection rule for and Equation 11.76. Hint: Sandwich each commutator between and .
       ∗∗ Problem 11.15 Obtain the selection rule for   as follows:
(a)
Derive the commutation relation
Hint: First show that
Use this, and (in the final step) the fact that demonstrate that
The generalization from z to r is trivial. Sandwich this commutator between
the implications.
     (b)
(11.78)
, to
and   , and work out
 ∗∗ Problem 11.16 An electron in the   , ,
by a sequence of (electric dipole) transitions to the ground state.
  (a)
(b) (c)
state of hydrogen decays What decay routes are open to it? Specify them in the following way:
If you had a bottle full of atoms in this state, what fraction of them would decay via each route?
What is the lifetime of this state? Hint: Once it’s made the first transition, it’s no longer in the state   , so only the first step in each sequence is relevant in computing the lifetime.
 537

11.4 Fermi’s Golden Rule
In the previous sections we considered transitions between two discrete energy states, such as two bound states of an atom. We saw that such a transition was most likely when the final energy satisfied the resonance condition: , where ω is the frequency associated with the perturbation. I now want to look at the case where   falls in a continuum of states (Figure 11.10). To stick close to the example of Section 11.2, if the radiation is energetic enough it can ionize the atom—the photoelectric effect—exciting the electron from a bound state into the continuum of scattering states.
Figure 11.10: A transition (a) between two discrete states and (b) between a discrete state and a continuum of states.
We can’t talk about a transition to a precise state in that continuum (any more than we can talk about someone being precisely 16 years old), but we can compute the probability that the system makes a transition to a state with an energy in some finite range   about   . That is given by the integral of Equation 11.35 over all the final states:
(11.79)
where . The quantity   is the number of states with energy between E and ;   is called the density of states, and I’ll show you how it’s calculated in Example 11.2.
At short times, Equation 11.79 leads to a transition probability proportional to   , just as for a transition between discrete states. On the other hand, at long times the quantity in curly brackets in Equation 11.79 is sharply peaked: as a function of   its maximum occurs at   and the central peak has a width of   . For sufficiently large t, we can therefore approximate Equation 11.79 as23
           The remaining integral was already evaluated in Section 11.2.3:
The oscillatory behavior of P has again been “washed out,” giving a constant transition rate:24 538
(11.80)
  
 Equation 11.81 is known as Fermi’s Golden Rule.25 Apart from the factor of   , it says that the transition rate is the square of the matrix element (this encapsulates all the relevant information about the dynamics of the process) times the density of states (how many final states are accessible, given the energy supplied by the perturbation—the more roads are open, the faster the traffic will flow). It makes sense.
Example 11.2
Use Fermi’s Golden Rule to obtain the differential scattering cross-section for a particle of mass m and incident wave vector   scattering from a potential (Figure 11.11).
Figure11.11: Aparticlewithincidentwavevector isscatteredintoastatewithwavevectork. Solution:
We take our initial and final states to be plane waves:
(11.82)
Here I’ve used a technique called box normalization; I place the whole setup inside a box of length l on a side. This makes the free-particle states normalizable and countable. Formally, we want the limit
; in practice l will drop out of our final expression. Using periodic boundary conditions,26 the allowed values of are
(11.83)
for integers   , , and   . Our pertu

"""

# insert into vectordb
added = add_to_vectordb(TASK_TEXT)
print(f"added to db: {added}")


# # Init LLM
# llm = OpenAIChat(
#     openai_api_key=""
# )

# Query vectordb
query = "What are einsteins coefficients?"
task = str(query_vectordb(query)["documents"][0])
print(f"task: {task}")

# # # Send the query back into the llm
# response = llm(task)
# print(response)
