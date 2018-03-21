val batch = MNIST.buildBatch(settings.testBatchSize)
/**
 * not used
 * 0 -> 01234_01, 01789_07, 01345_04, 03478_03, 03489_0
 * 1 -> 01234_01, 89012_19, 12345_12, 01456_14
 * 2 -> 12345_12
 * 4 -> 90123_19, 34589_4
 * 6 -> 67890_6
 * 7 -> 01678_07, 01567_17, 01567_17
 */
val detectorNames = mapOf(
        0 to listOf("01234_0", "01678_07", "02345_04"),
        1 to listOf("12345_124", "78901_1", "90123_19"),
        2 to listOf("23456_256", "02456_2", "23456_2", "12345_124"),
        3 to listOf("03478_03", "34567_3", "36789_378"),
        4 to listOf("12345_124", "01456_14", "02345_04"),
        5 to listOf("23456_256", "45678_5", "24589_59"),
        6 to listOf("23456_256", "34567_6", "56789_6"),
        7 to listOf("01789_07", "35678_7", "36789_378"),
        8 to listOf("36789_378", "45689_8", "03478_03"),
        9 to listOf("89012_19", "90123_19", "23589_9")
)

private fun testMultiNets() {
    var counter = 0
    val detectors = detectorNames.map { detector ->
        detector.key to detector.value.stream().map {
            NetworkIO().load("nets/nw$it.net")!!
        }.toArray().map { it as Network }
    }
    batch.stream().forEach { image ->
        val d = detectors.map { detector ->
            detector.first to detector.second.stream().map {
                it.activate(image)[detector.first]
            }.toArray().map { it as Double }.average()
        }
        val r = d.maxBy { it.second }!!.first
        if (r == image.index) {
            counter++
            //println("${image.index} - ok")
        } else {
            println("${image.index} -> $r [${d.map { it.first to (it.second * 1000).toInt() / 10.0 }}]")
        }

    }
    println("$counter / ${batch.size}")
}

fun main(args: Array<String>) {
    testMultiNets()
//    filterBad(2)
}

private fun filterBad(n: Int) {
    val nets = detectorNames[n]!!.map { it to NetworkIO().load("nets/nw$it.net")!! }
    var goodAnswer = 0
    var badAnswer = 0
    nets.forEach { (name, nw) ->
        print(name)
        batch.forEach {
            val r = nw.activate(it)
            if (it.index == n) {
                if (r[n] > 0.5) goodAnswer++
                else badAnswer++
            } else {
                if (r[n] > 0.5) badAnswer++
            }
        }
        println(" -> $goodAnswer / $badAnswer = ${(goodAnswer * 1000.0 / (goodAnswer + badAnswer)).toInt() / 10.0}")
    }
}