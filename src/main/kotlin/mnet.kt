fun main(args: Array<String>) {
    val batch = MNIST.buildBatch(settings.testBatchSize)//.filter { it.index in listOf(0, 1, 2, 4, 5, 6) }
    val detectorNames = mapOf(
            0 to listOf("01234_01", "03478_03", "01234_0", "01345_04", "01678_07", "01789_07", "02345_04", "03489_0"),
            1 to listOf("01234_01", "12345_14", "78901_1", "89012_19", "90123_19", "01456_14", "12345_12"),
            2 to listOf("23456_256", "02456_2", "12345_12", "23456_2"),
            3 to listOf("03478_03", "34567_3", "36789_378"),
            4 to listOf("12345_14", "34589_4", "90123_19", "01456_14", "02345_04"),
            5 to listOf("23456_256", "45678_5"),
            6 to listOf("23456_256", "34567_6", "56789_6", "67890_6"),
            7 to listOf("01567_17", "01678_07", "01567_17", "01789_07", "35678_7", "36789_378"),
            8 to listOf("36789_378"),
            9 to listOf("89012_19", "90123_19")
    )
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
            println("${image.index} -> $r [${d.map { it.first to (it.second*1000).toInt()/10.0 }}]")
        }

    }
    println("$counter / ${batch.size }")
}