# GameBench: Benchmarking LLMs via Games 

**GameBench** is a project to benchmark LLMs via a set of games. Why? Benchmarking via games has advantages over traditional knowledge-based benchmarks (MMLU, GPQA, Coding Benchmarks, etc.)

* Simultaneously lower skill floor and higher skill ceiling. Many benchmarks suffer from a short life-span: they debut with frontier models scoring 5-20%, only to be quickly hill-climbed to 80-90% and then lose the capability to discern performance differentials. Game Benchmarks on the other hand can often be used to differentiate between models many generations apart in a meaningful manner—something very difficult for traditional benchmarks to do.

* Lack of gameability: Ironically, Game Benchmarks are the least gameable. While one can train directly on the test set for knowledge benchmarks, one would have to at least set-up an RL environment with multiple different games to meaningfully improve their score on game benchmarks. 

* Contamination-Proof: Similar to the above, Game Benchmarks are more or less contamination proof. Many of the games people benchmark LLMs on are games that LLMs are already aware of and they likely have any strategy information on how to play them in their training data already.

* Extensibility: Compared to more traditional benchmarks, game benchmarks can easily be made harder (or easier) by simple extensions. For example, one could test adaptability by having two model's repeatedly play other and carry over context to understand if models are capable of adapting and exploiting weaknesses in the other's gameplay. Or one could easily take a game that can support multiple players and modify the benchmark by increasing (or decreasing) the # of players—thereby requiring models to deal with a more (or less) complex game state.

* Cost of Creation: As frontier models becoming increasingly capable, new challenging knowledge based benchmarks are only capable of being created by the world's most leading experts meaning that any new questions or benchmarks are extremely resource-intensive to produce. 