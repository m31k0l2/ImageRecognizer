import java.io.File
import java.util.*
import java.util.stream.Collectors
import kotlin.math.max
import kotlin.math.min

/**
 * Модель особи.
 * [nw] - нейронная сеть, [rate] - рейтинг выживаемости
 */
data class Individual(val nw: Network, var rate: Double=1.0) {
    fun rate(batch: List<Image>, rateCount: Int, alpha: Double) {
        val b = batch.groupBy { it.index }.map {
            (i, list) ->  i to list.shuffled().take(rateCount)
        }.toMap()
        val numbers = b.keys.toSortedSet()
        rate = b.keys.map { i ->
            val n = numbers.indexOf(i)
            b[i]!!.map { nw.activate(it, alpha) }.map { (1 - it[n])*(1 - it[n]) }.average()
        }.average()
    }
}

/**
 * [populationSize] - размер популяции
 * [scale] - предельное значение гена (веса сети). Вес выбирается из диапазона [-scale; +scale]
 * [crossoverRate] - параметр регулирующий передачу генов от родителей при скрещивании
 * 0,5 - половина от отца, половина от матери
 * 0,3 - 30 % от отца, 70 % от матери
 * 1,0 - все гены от матери
 * [mutantRate] - вероятность мутации генов, не должно быть сильно большим значением, основную роль в эволюции
 * должно играть скрещивание
 */
abstract class NetEvolution(
        var maxMutateRate: Double=0.2,
        private val maxRateCount: Int = 3,
        private val scale: Int=1
) {
    private val random = Random()
    private var rateCount = 1
    var mutantRate = 1.0
    var genMutateRate = 0.05
    var name: String = "nets/nw.net"
    lateinit var mutantStrategy: (epoch: Int, epochSize: Int) -> Double
    var leader: Individual? = null
    lateinit var batch: List<Image>
    var trainLayers = emptyList<Int>()
    private var mutateRate = 0.1
    private val populationAdder = 10
    var minPopulationSize = 60
    private var alpha: Double = 1.0

    init {
        if (File("nets/").mkdir()) {
            println("Создаю каталог nets/")
        }
    }

    /**
     * Запуск эволюции.
     * [epochSize] - количество поколений которые участвуют в эволюции (эпох).
     * Создаём популяцию размером populationSize
     * Выполняем эволюцию популяции от эпохи к эпохе
     */
    fun evolute (epochSize: Int, populationSize: Int, alpha: Double, maxStagnation: Int=5) = evolute(epochSize, generatePopulation(populationSize, name), alpha, maxStagnation)

    fun evolute(epochSize: Int, initPopulation: List<Individual>, alpha: Double, maxStagnation: Int=5): List<Individual> {
        if (trainLayers.isEmpty()) trainLayers = (0 until initPopulation.first().nw.layers.size).toList()
        this.alpha = alpha
        var population = initPopulation
        population.forEach { it.rate = 1.0 }
        var stagnation = 0
        var lastRate = 0.0
        mutateRate = max((Random().nextDouble()*maxMutateRate*100).toInt()/100.0, 0.005)
        (0 until epochSize).forEach {curEpoch ->
            println("эпоха $curEpoch")
            val start = System.nanoTime()
            mutantRate = mutantStrategy(curEpoch, epochSize)
            population = evoluteEpoch(population)
            val fin = System.nanoTime()
            val getRate = { pos: Int, population: List<Individual> -> ((population[pos].rate*1000000).toInt()/10000.0).toString()}
            println("мутация ${(mutantRate*1000).toInt()/10.0} % / $mutateRate")
            println("популяция ${population.size}")
            if (population.size > minPopulationSize) {
                val rateInfo = "${getRate(0, population)} < ${getRate(population.size / 4 - 1, population)} < ${getRate(population.size / 2 - 1, population)}"
                println("Рейтинг [$rateCount]: $rateInfo")
            } else {
                println("Рейтинг [$rateCount]: ${getRate(0, population)}")
            }
            println("Время: ${(fin-start)/1_000_000} мс\n")
            leader = population.first()
            val median = population[population.size/2-1]
            if (lastRate == median.rate) {
                stagnation++
                population = if (population.size < 120) {
                    population.union(generatePopulation(populationAdder, leader!!)).toList()
                } else {
                    population.take(minPopulationSize)
                }
                ratePopulation(population)
                population = population.sortedBy { it.rate }
                leader = population.first()
            } else {
                lastRate = median.rate
            }
            if (leader!!.rate < 0.02) {
                rateCount++
                population = population.take(minPopulationSize)
                ratePopulation(population.filter { it.rate < 0.3 })
            }
            if (rateCount > maxRateCount || stagnation == 5*maxStagnation) return population
        }
        return population
    }

    /**
     * эволюция популяции за одну эпоху
     * среди популяции проводим соревнование
     * По итогам генерируется новое поколение для следующей эпохи
     * Условия размножения (mutantRate и crossoverRate) изменяются от эпохи к эпохе
     **/

    private fun evoluteEpoch(initPopulation: List<Individual>): List<Individual> {
        val population = competition(initPopulation)
        return nextGeneration(population)
    }

    /**
     * Создаёт популяцию особей заданного размера [size]
     */
    private fun generatePopulation(size: Int) = (0 until size).map { createIndividual() }

    private fun generatePopulation(size: Int, name: String): List<Individual> {
        leader?.let {
            it.rate = 1.0
            return generatePopulation(size, it)
        }
        if (!File(name).exists()) return generatePopulation(size)
        val nw = CNetwork().load(name)!!
        return generatePopulation(size, Individual(nw))
    }

    private fun generatePopulation(size: Int, individual: Individual) = (0 until size).map {
        if (it == 0) individual else Individual(individual.nw.clone().dropout(trainLayers, 0.5, true))
    }

    private fun createIndividual() = Individual(createNet())

    /**
     * Задаётся топология сети
     */
    abstract fun createNet(): Network

    /**
     * Проводит соревнование внутри популяции. На выходе вычисляет поколение
     */
    private fun competition(population: List<Individual>): List<Individual> {
        ratePopulation(population.filter { it.rate == 1.0 })
        return population.sortedBy { it.rate }
    }

    private fun ratePopulation(population: List<Individual>) = population.parallelStream().forEach { individ ->
        individ.rate(batch, rateCount, alpha)
    }

    /**
     * Выводим новое поколение популяции.
     * Для начала убиваем половину самых слабых особей. Восстановление популяции произойдёт за счёт размножения
     * Потом выживших случайным образом разбиваем на пары.
     * Скрещивание будет происходить беспорядочно, т.е. не моногамно, каждая особь может быть в паре с любой
     * другой особью несколько раз, а количество пар будет равно количеству популяции
     * Пары скрещиваем и получаем потомство. Скрещивание выполняется многопоточно.
     * Одна пара рождает одного ребёнка, таким образом получаем, что в итоге количество скрещивающихся особей будет
     * соответствовать рождённым особям.
     * Из числа родителей и потомства составляем новое поколение популяции
     * Для исключения вырождения популяции можем удваивать на время мутацию. Под вырождением понимаем критическое
     * совпадение генов у всех особей популяции. Это означает, что созданный шаблон при исходных наборах гена оптимален.
     * Этот оптимальный шаблон сохраним, для дальнейшего сравнения и использования
     */
    private fun nextGeneration(population: List<Individual>): List<Individual> {
        val survivors = if (population.size < minPopulationSize) population else population.take(population.size/2)
        val mutants = survivors.mapNotNull { if (random.nextDouble() < mutantRate) mutate(it) else null }.take(population.size/2)
        val parents = selection(survivors, population.size/2-mutants.size)
        val offspring = createChildren(parents)
        return survivors.union(offspring).union(mutants).toList()
    }

    private fun createChildren(parents: List<Pair<Individual, Individual>>) = parents.parallelStream().map { cross(it) }.collect(Collectors.toList())!!

    /**
     * Разбиваем популяцию по парам для их участия в скрещивании
     */
    private fun selection(population: List<Individual>, count: Int): List<Pair<Individual, Individual>> {
        val size = population.size
        val s = size*(size + 1.0)
        var rangs = population.asReversed().mapIndexed { index, individual ->
            individual to 2*(index+1)/s
        }
        var rangCounter = 0.0
        rangs = rangs.map {
            rangCounter += it.second
            it.first to rangCounter
        }
        return (0 until count).map {
            rangs.find { it.second > random.nextDouble() }!!.first to rangs.find { it.second > random.nextDouble() }!!.first
        }
    }

    private fun getCrossoverRate(parents: Pair<Individual, Individual>) = if (parents.first.rate < parents.second.rate) {
        0.5 + 0.2*random.nextDouble()
    } else {
        0.5 - 0.2*random.nextDouble()
    }

    private fun cross(parents: Pair<Individual, Individual>): Individual {
        val crossoverRate = getCrossoverRate(parents)
        val nw = parents.first.nw.clone()
        trainLayers.forEach { l ->
            val layer = nw.layers[l]
            val neurons = layer.neurons
            for (i in 0 until neurons.size) {
                if (random.nextDouble() > crossoverRate) {
                    neurons[i] = parents.second.nw.layers[l].neurons[i]
                }
            }
        }
        return Individual(nw)
    }

    private fun mutate(individual: Individual): Individual? {
        var isMutate = false
        val nw = individual.nw.clone()
        for (l in trainLayers) {
            val layer = nw.layers[l]
            for (neuron in layer.neurons) {
                if (random.nextDouble() < mutateRate) {
                    isMutate = true
                    val weights = neuron.weights
                    val n = Random().nextInt(max(1.0, genMutateRate * weights.size).toInt())
                    for (i in (0..n)) {
                        weights[Random().nextInt(weights.size)] = (1 - 2 * random.nextDouble()) * scale
                    }
                }
            }
        }
        return Individual(nw).takeIf { isMutate }
    }
}