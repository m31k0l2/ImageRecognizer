open class Agent(private val nets: List<Network>) {
    fun recognize(image: Image): Int? {
        val results = nets.map { it.activate(image, 15.0)[0] }
        if (!results.any { it > 0.5 }) return null
        if (results.filter { it > 0.5 }.size > 1) return null
        return results.indexOfFirst { it > 0.5 }
    }
}

class TestAgent: Agent((0..number).map { CNetwork().load("nets/nw$it.net")!! })

class Agent0: Agent((0..9).map { CNetwork().load("nets/agent0/nw$it.net")!! })

fun main(args: Array<String>) {
    var counter = 0
    val batch = MNIST.buildBatch(500).filter { it.index in (0..number) }
    val agent = TestAgent()
    batch.forEach { image ->
        val r = agent.recognize(image)
        println("${image.index} -> $r")
        if (r != image.index) counter++
    }
    println("${batch.size - counter}/${batch.size}")
}