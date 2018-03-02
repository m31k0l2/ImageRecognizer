import java.io.File
import java.io.FileWriter
import java.util.*
import java.util.stream.Collectors
import kotlin.math.max


/**
 * Модель особи.
 * [nw] - нейронная сеть, [rate] - рейтинг выживаемости
 */
data class Individual(val nw: Network, var rate: Double=1.0)

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
class ImageNetEvolution(
        private val layers: List<Int>,
        var mutantGenRate: Double=.005,
        private val scale: Int=1
) {
    private val random = Random()
    var mutantRate = 1.0
    lateinit var mutantStrategy: (epoch: Int, epochSize: Int) -> Double
    private var leader: Individual? = null
    lateinit var batch: List<Image>

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
    fun evolute (epochSize: Int, populationSize: Int) = evolute(epochSize, generatePopulation(populationSize, "nw"))

    fun evolute(epochSize: Int, initPopulation: List<Individual>, maxStagnation: Int=5): List<Individual> {
        var population = initPopulation
        var stagnation = 0
        var lastRate = 0.0
        (0 until epochSize).forEach {curEpoch ->
            println("эпоха $curEpoch")
            val start = System.nanoTime()
            mutantRate = mutantStrategy(curEpoch, epochSize)
            population = evoluteEpoch(population)
            val fin = System.nanoTime()
            val getRate = { pos: Int, population: List<Individual> -> ((population[pos].rate*1000000).toInt()/10000.0).toString()}
            val rateInfo = "${getRate(0, population)} < ${getRate(population.size/4-1, population)} < ${getRate(population.size/2-1, population)}"
            println("мутация ${(mutantRate*1000).toInt()/10.0} %")
            println("Размер батча ${batch.size}")
            println("Рейтинг $rateInfo")
            println("Время: ${(fin-start)/1_000_000} мс\n")
            leader = population.first()
            val median = population[population.size/2-1]
            if (lastRate == median.rate) {
                stagnation++
            } else {
                lastRate = median.rate
                stagnation = 0
            }
            if (stagnation > maxStagnation || leader!!.rate > 0.999*median.rate || leader!!.rate < .01) return population
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

    private fun writeToFile(text: String) {
        FileWriter("stat.txt", true).use { writer ->
            // запись всей строки
            writer.write(text + '\n')
            writer.flush()
        }
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
        if (!File("nets/$name.net").exists()) return generatePopulation(size)
        val nw = NetworkIO().load("nets/$name.net")!!
        return generatePopulation(size, Individual(nw))
    }

    private fun generatePopulation(size: Int, individual: Individual) = (0 until size).map {
        if (it == 0) individual else mutate(individual) ?: createIndividual()
    }

    private fun createIndividual() = Individual(createNet())

    /**
     * Задаётся топология сети
     */
    private fun createNet() = Network(*layers.toIntArray())

    /**
     * Проводит соревнование внутри популяции. На выходе вычисляет поколение
     */
    private fun competition(population: List<Individual>): List<Individual> {
        ratePopulation(population.filter { it.rate == 1.0 })
        return population.sortedBy { it.rate }
    }

    private fun ratePopulation(population: List<Individual>) = population.parallelStream().forEach {
        it.rate = 1 - batch.map { image ->
            it.nw.activate(image.colorsMatrix)[image.index]
        }.average()
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
        val survivors = population.take(population.size/2)
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

    /**
     * Выполняем скрещивание.
     * Гены потомка получаются либо путём мутации, либо путём скрещивания (определяется случайностью)
     * Если гены формируются мутацией, то значение гена выбирается случайно в диапазоне [-scale; scale]
     * Если гены формируются скрещиванием, то ген наследуется случайно от одного из родителей
     * Шанс передачи гена от родителя определяется параметром crossoverRate
     * Гены нейронной сети - это веса её нейронов
     */
    private fun cross(pair: Pair<Individual, Individual>): Individual {
        val firstParent = pair.first.nw
        val secondParent = pair.second.nw
        val firstParentGens = extractWeights(firstParent)
        val secondParentGens = extractWeights(secondParent)
        var crossoverRate = .5
        if (pair.first.rate < pair.second.rate) {
            crossoverRate += 0.2*random.nextDouble()
        } else {
            crossoverRate -= 0.2*random.nextDouble()
        }
        val childGens = firstParentGens.mapIndexed { l, layer ->
            layer.mapIndexed { n, neuron -> neuron.mapIndexed { w, gen ->
                gen.takeIf { random.nextDouble() < crossoverRate } ?: secondParentGens[l][n][w] }
            }
        }
        return Individual(generateNet(childGens))
    }

    private fun mutate(individual: Individual): Individual? {
        val genMutateRate = max(random.nextDouble()*0.1, mutantGenRate)
        val parentGens = extractWeights(individual.nw)
        var isMutate = false
        val childGens = parentGens.map { layer ->
            layer.map { neuron ->
                neuron.map { gen ->
                    if (random.nextDouble() < genMutateRate) {
                        isMutate = true
                        (1 - 2 * random.nextDouble()) * scale
                    } else gen
                }
            }
        }
        return Individual(generateNet(childGens)).takeIf { isMutate }
    }

    /**
     * Создаёт нейронную сеть на основе списка весов, упакованных следующим образом:
     * [ уровень_слоя [ уровень_нейрона [ уровень_веса ] ]
     * Проходим по каждому уровню и заполняем сеть
     */
    private fun generateNet(layerWeights: List<List<List<Double>>>): Network {
        val nw = createNet()
        layerWeights.forEachIndexed { layerPosition, neuronsWeights ->
            val layer = nw.layers[layerPosition]
            layer.neurons.forEachIndexed { index, neuron ->
                neuron.weights = neuronsWeights[index].toMutableList()
            }
        }
        return nw
    }

    /** Извлекаем веса нейронной сети и упаковываем их специальным образом */
    private fun extractWeights(nw: Network) = nw.layers.map { it.neurons.map { it.weights.toList() } }

    /** определяет усреднённое различие между сетями **/
    private fun netsDifferent(nets: List<Network>): Double {
        val extractWeights: (Network) -> List<Double> = {
            nw: Network -> nw.layers.flatMap { it.neurons }.flatMap { it.weights }
        }
        val bestWeights = extractWeights(nets.first())
        val difs = mutableListOf<Double>()
        (1 until nets.size).map { nets[it] }.forEach {
            val weights = extractWeights(it)
            var difsCount = 0
            weights.forEachIndexed { i, w ->
                if (w != bestWeights[i]) difsCount++
            }
            difs.add(difsCount*1.0/bestWeights.size)
        }
        return difs.average()
    }

    fun dropout(population: List<Individual>, p: Double): List<Individual> {
        val generation = population.map { it.nw }.map { it.clone() }.map { Individual(it) }
        generation.parallelStream().forEach {
            it.nw.layers.forEach {
                it.neurons.forEach {
                    it.weights.forEachIndexed { index, d ->
                        it.weights[index] = if (random.nextDouble() < p) .0 else d
                    }
                }
            }
        }
        return competition(generation.union(population).toList()).take(population.size/2)
    }
}