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

private fun Agent.test(showError: Boolean) {
    var counter = 0
    var errors = 0
    val batch = MNIST.buildBatch(500).filter { it.index in (0..9) }//6,9
    batch.forEach { image ->
        try {
            val r = recognize(image)
            if (r != image.index) {
                counter++
                if (showError) println("${image.index} -> !! $r")
                errors++
            } else {
                println("${image.index} -> $r")
            }
        } catch (e: RecognizeError) {
            counter++
            if (e is ManyRecognizeError) {
                if (showError) println("${image.index} -> ?? ${e.results}")
            } else {
                if (showError) println("${image.index} -> ?")
            }
        }
    }
    println("${batch.size - counter} [!!$errors]/${batch.size}")
}

fun main(args: Array<String>) {
    TestAgent().test(true)
}