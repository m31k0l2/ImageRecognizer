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
        private var populationSize: Int,
        private val scale: Int,
        private val initMutantRate: Double=.1
) {
    private val random = Random()
    var mutantRate = initMutantRate

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
    fun evolute(epochSize: Int): Network {
        var population = generatePopulation(populationSize, "nw")
        var batchSize = 10
        var nextChange = batchSize * 2
        var lastResult = 0.0
        var stagnation = 0
        MNIST.createBatch(batchSize)
        (0 until 2*epochSize).forEach {curEpoch ->
            if (stagnation == 10) {
                return population.first().nw
            }
            println("эпоха $curEpoch")
            val start = System.nanoTime()
            mutantRate = ((epochSize - curEpoch)*1.0/epochSize).takeIf { it > 0 } ?: 0.01
            if (population.first().rate < .1) {
                return population.first().nw
            }
            population = evoluteEpoch(population)
            val fin = System.nanoTime()
            val getRate = { pos: Int, population: List<Individual> -> ((population[pos].rate*1000000).toInt()/10000.0).toString()}
            if (curEpoch < epochSize && curEpoch % 10 == 0) {
                population = rateGeneration(batchSize, population)
            } else if (curEpoch > epochSize && batchSize == 10){
                batchSize = 40
                population = rateGeneration(batchSize, population)
            }
            val curResult = population[populationSize/2-1].rate
            if (curResult == lastResult) {
                stagnation++
                MNIST.createBatch(batchSize)
            } else {
                lastResult = curResult
            }
            val rateInfo = "${getRate(0, population)} < ${getRate(populationSize/4-1, population)} < ${getRate(populationSize/2-1, population)}"
            writeToFile(getRate(0, population).replace(".", ","))
            println("Рейтинг $rateInfo")
            println("Размер батча: $batchSize")
            if (population.first().rate*1.001 > population[populationSize/2-1].rate) return population.first().nw
            println("Время: ${(fin-start)/1_000_000} мс\n")
        }
        return population.first().nw
    }

    private fun rateGeneration(batchSize: Int, population: List<Individual>): List<Individual> {
        var population1 = population
        MNIST.createBatch(batchSize)
        rateGeneration(population1)
        population1 = population1.sortedBy { it.rate }
        return population1
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
    private fun generatePopulation(size: Int=populationSize) = (0 until size).map { createIndividual() }

    private fun generatePopulation(size: Int=populationSize, name: String): List<Individual> {
        if (!File("nets/$name.net").exists()) return generatePopulation(size)
        val nw = NetworkIO().load("nets/$name.net")!!
        return generatePopulation(size, Individual(nw))
    }

    private fun generatePopulation(size: Int=populationSize, individual: Individual) = (0 until size).map {
        if (it == 0) individual else createIndividual()
    }

    private fun createIndividual() = Individual(createNet())

    /**
     * Задаётся топология сети
     */
    private fun createNet() = Network(8, 12, 160, 40, 10)

    /**
     * Проводит соревнование внутри популяции. На выходе вычисляет поколение
     */
    private fun competition(population: List<Individual>): List<Individual> {
        rateGeneration(population.filter { it.rate == 1.0 })
        return population.sortedBy { it.rate }
    }

    private fun rateGeneration(population: List<Individual>) = population.parallelStream().forEach {
        it.rate = 1 - MNIST.batch.map { image ->
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
        val survivors = population.take(populationSize / 2)
        val mutants = survivors.mapNotNull { if (random.nextDouble() < mutantRate) mutate(it) else null }
        val parents = selection(survivors).take(survivors.size - mutants.size)
        val offspring = createChildren(parents)
        val generation = survivors.union(offspring).union(mutants).toList()
//        val dif = netsDifferent(generation.map { it.nw })
//        mutantRate = initMutantRate + max((1 - dif*8) *mutantRate, .0)
        println("мутация ${(mutantRate*1000).toInt()/10.0} %")
        return generation
    }

    private fun createChildren(parents: List<Pair<Individual, Individual>>) = parents.parallelStream().map { cross(it) }.collect(Collectors.toList())!!

    /**
     * Разбиваем популяцию по парам для их участия в скрещивании
     */
    private fun selection(population: List<Individual>): List<Pair<Individual, Individual>> {
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
        return (0 until population.size).map {
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
        val genMutateRate = max(random.nextDouble()*0.1, 0.005)
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

    private fun dropout(population: List<Individual>, p: Double=initMutantRate): List<Individual> {
        val generation = population.map { it.nw }.map { it.clone() }.map { Individual(it) }
        population.parallelStream().forEach {
            it.rate = 1.0
            it.nw.layers.forEach {
                it.neurons.forEach {
                    it.weights.forEachIndexed { index, d ->
                        it.weights[index] = if (random.nextDouble() < p) .0 else d
                    }
                }
            }
        }
        return generation
    }
}