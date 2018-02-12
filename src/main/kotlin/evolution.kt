import java.util.*
import java.util.concurrent.atomic.AtomicInteger
import java.util.stream.Collectors

/**
 * Модель особи.
 * [nw] - нейронная сеть, [rate] - рейтинг выживаемости
 */
data class Individual(val nw: Network, var rate: AtomicInteger= AtomicInteger(0))

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
abstract class AbstractEvolution(
        private var populationSize: Int,
        private val scale: Int,
        private val initMutantRate: Double=0.1
) {
    private val random = Random()
    var mutantRate = initMutantRate
    var curEpoch = 0

    /**
     * Запуск эволюции.
     * [epochSize] - количество поколений которые участвуют в эволюции (эпох).
     * Создаём популяцию размером populationSize
     * Выполняем эволюцию популяции от эпохи к эпохе
     */
    fun evolute(epochSize: Int, batchSize: Int, period: Int): Individual {
        var population = generatePopulation()
        var dropoutCounter = 0
        var dropoutRate = population[populationSize/2-1].rate.get()
        (0 until epochSize).forEach {
            curEpoch = it
            if (curEpoch % period == 0) {
                if (curEpoch != 0) {
                    val leader = population.first()
                    if (MNIST.batch.isNotEmpty()) testNet(leader.nw)
                    population = generatePopulationFrom(leader, populationSize)
                }
                MNIST.createBatch(batchSize)
            }
            population = evoluteEpoch(population)
            val testRate = population[populationSize/2-1].rate.get()
            if (dropoutRate != testRate) {
                dropoutRate = testRate
                dropoutCounter = 0
            }
            else dropoutCounter++
            if (dropoutCounter == 5) {
                population.forEach { it.rate.set(0) }
                dropout(population, mutantRate)
            }
        }
        return population.first()
    }

    /**
     * эволюция популяции за одну эпоху
     * среди популяции проводим соревнование
     * По итогам генерируется новое поколение для следующей эпохи
     * Условия размножения (mutantRate и crossoverRate) изменяются от эпохи к эпохе
     **/
    open fun evoluteEpoch(initPopulation: List<Individual>): List<Individual> {
        val population = competition(initPopulation)
        val getRate = { pos: Int -> (population[pos].rate.get()/(10.0*MNIST.batch.size)).toString()}
        println("Рейтинг ${getRate(0)} < ${getRate(populationSize/4-1)} < ${getRate(populationSize/2-1)}")
        return nextGeneration(population) // генерируем следующее поколение особей
    }

    /**
     * Создаёт популяцию особей заданного размера [size]
     */
    open fun generatePopulation(size: Int=populationSize) = (0 until size).map { createIndividual() }

    fun generatePopulationFrom(individual: Individual, size: Int=populationSize): List<Individual> {
        val weights = extractWeights(individual.nw)
        individual.rate.set(0)
        return (0 until size).map {
            if (it == 0) individual
            else {
                val mutantWeights = weights.map { it.map { it.map { if (random.nextDouble() < mutantRate) random.nextDouble() else it } } }
                Individual(generateNet(mutantWeights))
            }
        }
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
        val newGeneration = population.filter { it.rate.get() == 0 }
        MNIST.batch.forEach { image ->
            newGeneration.parallelStream().forEach {
                val o = it.nw.activate(image.colorsMatrix)
                val result = o[image.index]
                val error = 1 - result
                it.rate.addAndGet((error*error*1000).toInt())
            }
        }
        return population.sortedBy { it.rate.get() }
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
    open fun nextGeneration(population: List<Individual>): List<Individual> {
        val survivors = population.take(populationSize / 2)
        val parents = selection(survivors)
        val generation = survivors.union(createChildren(parents)).toList()
        val dif = netsDifferent(generation.map { it.nw })
        mutantRate = when {
            dif < .05 -> initMutantRate*2
            dif > .3 -> initMutantRate*.5
            else -> initMutantRate
        }
        inform(dif) // информируем терминал об изменениях
        return generation
    }

    private fun createChildren(parents: List<Pair<Individual, Individual>>) = parents.parallelStream().map { cross(it) }.collect(Collectors.toList())!!

    open fun inform(dif: Double) {
        println("различие ${(dif*1000).toInt() / 10} %")
        println("мутация ${(mutantRate*1000).toInt()/10.0} %")
    }

    /**
     * Разбиваем популяцию по парам для их участия в скрещивании
     */
    private fun selection(players: List<Individual>) = (1..populationSize/2).map {
        players[random.nextInt(players.size)] to players[random.nextInt(players.size)]
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
        var crossoverRate = 0.5
        if (pair.first.rate.get() > pair.second.rate.get()) {
            crossoverRate += 0.2*random.nextDouble()
        } else {
            crossoverRate -= 0.2*random.nextDouble()
        }
        val childGens = firstParentGens.mapIndexed { l, layer ->
            layer.mapIndexed { n, neuron -> neuron.mapIndexed { w, gen ->
                if (random.nextDouble() < mutantRate) {
                    (1 - 2*random.nextDouble())*scale
                } else gen.takeIf { random.nextDouble() < crossoverRate } ?: secondParentGens[l][n][w] }
            }
        }
        return Individual(generateNet(childGens))
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

    private fun dropout(population: List<Individual>, p: Double) = population.forEach {
        it.nw.layers.forEach {
            it.neurons.forEach {
                it.weights.forEachIndexed { index, d ->
                    it.weights[index] = if (random.nextDouble() < p) 0.0 else d
                }
            }
        }
    }
}