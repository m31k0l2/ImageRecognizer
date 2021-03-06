import kotlin.math.round

open class RecognizeError: Exception()
class ManyRecognizeError(val results: Map<Int, Double>): RecognizeError()
class NullRecognizeError: RecognizeError()

open class Agent(private val nets: List<Network>) {
    fun recognize(image: Image): Int {
        val results = nets.map { it.activate(image, 15.0)[0] }
        if (!results.any { it > 0.5 }) throw NullRecognizeError()
        if (results.filter { it > 0.5 }.size > 1) throw ManyRecognizeError(results.mapIndexed { i, d -> i to round(d) }.toMap())
        return results.indexOfFirst { it > 0.5 }
    }
}

class TestAgent: Agent((0..9).mapNotNull { CNetwork().load("nets/nw$it.net") })

class Agent0: Agent((0..9).mapNotNull { CNetwork().load("nets/agent0/nw$it.net") })
class Agent1: Agent((0..9).mapNotNull { CNetwork().load("nets/agent1/nw$it.net") })

private fun Agent.test(showError: Boolean) {
    var counter = 0
    var errors = 0
    val batch = MNIST.buildBatch(500).filter { it.index in (0..9) }
    val errorMap = mutableMapOf<Int, Int>()
    batch.forEach { image ->
        try {
            val r = recognize(image)
            if (r != image.index) {
                errorMap[image.index] = errorMap[image.index]?.plus(1) ?: 1
                counter++
                if (showError) println("${image.index} -> !! $r")
                errors++
            } else {
                println("${image.index} -> $r")
            }
        } catch (e: RecognizeError) {
            errorMap[image.index] = errorMap[image.index]?.plus(1) ?: 1
            counter++
            if (e is ManyRecognizeError) {
                if (showError) println("${image.index} -> ?? ${e.results}")
            } else {
                if (showError) println("${image.index} -> ?")
            }
        }
    }
    println("${batch.size - counter} [!!$errors]/${batch.size}")
    errorMap.toSortedMap().forEach {
        println(it)
    }
}

fun main(args: Array<String>) {
    Agent0().test(true)
}