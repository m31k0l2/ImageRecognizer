fun main(args: Array<String>) {
    val batch = MNIST.buildBatch(settings.testBatchSize)//.filter { it.index in listOf(0, 1, 2, 4, 5, 6) }
    val detectors = mapOf(
            0 to listOf("01234_01", "03478_03"),
            1 to listOf("01234_01", "12345_14", "78901_1", "89012_19", "90123_19"),
            2 to listOf("23456_256"),
            3 to listOf("03478_03"),
            4 to listOf("12345_14", "34589_4"),
            5 to listOf("23456_256", "45678_5"),
            6 to listOf("23456_256", "34567_6", "56789_6", "67890_6"),
            7 to listOf("78901_1"),
            8 to listOf("89012_19"),
            9 to listOf("89012_19", "90123_19")
    )
    var counter = 0
    batch.stream().forEach { image ->
        val d = detectors.map { detector ->
            detector.key to detector.value.map {
                val  nw = NetworkIO().load("nets/nw$it.net")!!
            nw.activate(image)[detector.key]
            }.average()
        }
        val r = d.maxBy { it.second }!!.first
        if (r == image.index) {
            counter++
            println("${image.index} - ok")
        } else {
            println("${image.index} -> $r [${d.map { it.first to (it.second*1000).toInt()/10.0 }}]")
        }

    }
    println("$counter / ${batch.size }")
}